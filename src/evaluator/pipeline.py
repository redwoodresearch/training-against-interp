from src import ChatMessage, MessageRole, Prompt, api
from src.docent_utils import AgentTranscript
from src.evaluator.classifier import label_transcript
from src.evaluator.ideas import generate_ideas
from src.model_organisms import BasicModel
from src.utils import gather_with_limits, get_project_root, load_prompt_file

PROMPTS_DIR = get_project_root() / "src" / "evaluator" / "prompts"
DEFAULT_MODEL = "x-ai/grok-4"


def _extract_tag(text: str, tag: str) -> str:
    start_tag, end_tag = f"<{tag}>", f"</{tag}>"
    if start_tag in text and end_tag in text:
        return text[text.find(start_tag) + len(start_tag) : text.find(end_tag)].strip()
    return ""


async def generate_plan(
    idea: str,
    behavior: str,
    scenario_type: str,
    num_turns: int,
    model_id: str,
    extra_instructions: str | None = None,
) -> dict:
    """
    Generate a conversation plan from a test idea.

    Args:
        idea: The test scenario idea
        behavior: Description of the behavior being tested
        scenario_type: Either "positive" or "negative"
        num_turns: Number of conversation turns to plan for
        model_id: Model to use for plan generation
        extra_instructions: Optional extra instructions for plan generation

    Returns:
        Dict with 'initial_message' and 'conversation_plan'
    """
    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=load_prompt_file(
                    PROMPTS_DIR / "plan.jinja2",
                    behavior=behavior,
                    idea=idea,
                    scenario_type=scenario_type,
                    num_turns=num_turns,
                    extra_instructions=extra_instructions,
                ),
            )
        ]
    )
    response = await api(
        model_id=model_id, prompt=prompt, temperature=1.0, max_tokens=4000
    )
    text = response[0].completion
    initial_message = _extract_tag(text, "initial_message")
    conversation_plan = _extract_tag(text, "conversation_plan")

    if not initial_message:
        print(
            f"[DEBUG] Plan generation returned no initial_message. Raw response (len={len(text)}):\n{repr(text[:500])}"
        )

    return {
        "initial_message": initial_message,
        "conversation_plan": conversation_plan,
    }


async def run_conversation(
    plan: dict,
    behavior: str,
    idea: str,
    scenario_type: str,
    model: BasicModel,
    num_turns: int,
    auditor_model_id: str,
) -> AgentTranscript:
    """
    Execute a multi-turn conversation between an auditor and the target model.

    Args:
        plan: Conversation plan with 'initial_message' and 'conversation_plan'
        behavior: Description of the behavior being tested
        idea: The test scenario idea
        scenario_type: Either "positive" or "negative"
        model: The model organism to evaluate
        num_turns: Number of conversation turns
        auditor_model_id: Model to use for the auditor

    Returns:
        AgentTranscript of the conversation
    """
    initial_message = plan["initial_message"]
    if not initial_message:
        raise ValueError(
            f"Plan generation failed - empty initial_message. Plan: {plan}"
        )

    auditor_system = load_prompt_file(
        PROMPTS_DIR / "auditor.jinja2",
        seed_prompt=initial_message,
        behavior=behavior,
        idea=idea,
        scenario_type=scenario_type,
        conversation_plan=plan["conversation_plan"],
    )

    transcript = AgentTranscript()
    target_messages = [ChatMessage(role=MessageRole.user, content=initial_message)]
    transcript.add_user_message(initial_message)

    for turn in range(num_turns):
        target_response = await model(Prompt(messages=target_messages))
        if not target_response or not target_response.strip():
            print(
                f"[DEBUG] Model returned empty response at turn {turn}. Response: {repr(target_response)}"
            )
            raise ValueError(f"Model returned empty response at turn {turn}")
        transcript.add_assistant_message(target_response)
        target_messages.append(
            ChatMessage(role=MessageRole.assistant, content=target_response)
        )

        if turn == num_turns - 1:
            break

        # Build auditor messages (flipped perspective)
        auditor_messages = [
            ChatMessage(role=MessageRole.system, content=auditor_system)
        ]
        for msg in transcript.messages:
            if msg.role == "user" and msg.content != initial_message:
                auditor_messages.append(
                    ChatMessage(role=MessageRole.assistant, content=msg.content)
                )
            elif msg.role == "assistant":
                auditor_messages.append(
                    ChatMessage(role=MessageRole.user, content=msg.content)
                )

        response = await api(
            model_id=auditor_model_id,
            prompt=Prompt(messages=auditor_messages),
            temperature=1.0,
            max_tokens=2000,
        )
        auditor_response = response[0].completion
        if not auditor_response or not auditor_response.strip():
            print(
                f"[DEBUG] Auditor returned empty response at turn {turn}. Response: {repr(auditor_response)}"
            )
            raise ValueError(f"Auditor returned empty response at turn {turn}")
        transcript.add_user_message(auditor_response)
        target_messages.append(
            ChatMessage(role=MessageRole.user, content=auditor_response)
        )

    return transcript


async def run_single_evaluation(
    idea: str,
    behavior: str,
    scenario_type: str,
    model: BasicModel,
    num_turns: int,
    model_id: str,
    extra_instructions: str | None = None,
) -> dict:
    """
    Run a single evaluation: plan -> conversation -> classification.

    Args:
        idea: The test scenario idea
        behavior: Description of the behavior being tested
        scenario_type: Either "positive" or "negative"
        model: The model organism to evaluate
        num_turns: Number of conversation turns
        model_id: Model for plan generation, auditor, and classification
        extra_instructions: Optional extra instructions passed to plan generation

    Returns:
        Dict with 'idea', 'scenario_type', 'plan', 'transcript', and 'label'
    """
    if not idea:
        raise ValueError("Empty idea passed to run_single_evaluation")
    if not behavior:
        raise ValueError("Empty behavior passed to run_single_evaluation")

    plan = await generate_plan(
        idea,
        behavior,
        scenario_type,
        num_turns,
        model_id,
        extra_instructions=extra_instructions,
    )
    transcript = await run_conversation(
        plan, behavior, idea, scenario_type, model, num_turns, model_id
    )
    label = await label_transcript(transcript, behavior, model_id)
    return {
        "idea": idea,
        "scenario_type": scenario_type,
        "plan": plan,
        "transcript": transcript,
        "label": label,
    }


async def run_evaluation(
    behavior: str,
    model: BasicModel,
    num_scenarios: int = 100,
    num_turns: int = 2,
    include_positive: bool = True,
    include_negative: bool = True,
    model_id: str = DEFAULT_MODEL,
    n_concurrents: int = 10,
) -> dict:
    """
    Run full evaluation pipeline: behavior -> ideas -> plans -> conversations -> scores.

    Args:
        behavior: Description of the behavior to test for
        model: The model organism to evaluate
        num_scenarios: Number of test scenarios to generate
        num_turns: Number of conversation turns per scenario
        include_positive: Whether to generate positive scenarios (behavior should manifest)
        include_negative: Whether to generate negative scenarios (behavior should not manifest)
        model_id: Model for idea generation, planning, auditor, and classification
        n_concurrents: Max concurrent API calls

    Returns:
        Dict with 'positive' and/or 'negative' keys, each containing list of evaluation results
    """
    ideas = await generate_ideas(
        behavior=behavior,
        num_scenarios=num_scenarios,
        include_positive=include_positive,
        include_negative=include_negative,
        model_id=model_id,
    )

    results = {}
    for category in ["positive", "negative"]:
        if category not in ideas:
            continue

        tasks = [
            run_single_evaluation(idea, behavior, category, model, num_turns, model_id)
            for idea in ideas[category]
        ]
        evaluations = await gather_with_limits(
            tasks,
            n_concurrents=n_concurrents,
            render_tqdm=True,
            tqdm_desc=f"Evaluating {category}",
        )
        results[category] = [e for e in evaluations if e is not None]

    return results
