import re

from tqdm.auto import tqdm

from src import ChatMessage, MessageRole, Prompt, api
from src.utils import get_project_root, load_prompt_file

PROMPTS_DIR = get_project_root() / "src" / "evaluator" / "prompts"


def _parse_scenarios(text: str) -> list[str]:
    """Parse XML scenarios from LLM output."""
    scenarios = []

    # Find all <scenario>...</scenario> blocks
    pattern = r"<scenario>(.*?)</scenario>"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        scenario = match.strip()
        if scenario:
            scenarios.append(scenario)

    return scenarios


async def _generate_chunk(
    behavior: str,
    scenario_type: str,
    num_scenarios: int,
    model_id: str,
    random_seed: int,
    trigger_conditions: str | None = None,
    extra_instructions: str | None = None,
) -> list[str]:
    """Generate a single chunk of scenarios."""
    content = f"idx: {random_seed}\n\n" + load_prompt_file(
        PROMPTS_DIR / "scenarios.jinja2",
        behavior=behavior,
        scenario_type=scenario_type,
        num_scenarios=num_scenarios,
        trigger_conditions=trigger_conditions,
        extra_instructions=extra_instructions,
    )

    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=content)])
    responses = await api(
        model_id=model_id, prompt=prompt, temperature=1.0, max_tokens=8000
    )

    return _parse_scenarios(responses[0].completion)


async def _generate(
    behavior: str,
    scenario_type: str,
    num_scenarios: int,
    model_id: str,
    chunk_size: int = 25,
    trigger_conditions: str | None = None,
    extra_instructions: str | None = None,
    tqdm_desc: str | None = None,
) -> list[str]:
    """Generate scenarios by sampling chunk_size at a time until we have enough."""
    scenarios = []
    seed = 0

    with tqdm(total=num_scenarios, desc=tqdm_desc) as pbar:
        while len(scenarios) < num_scenarios:
            chunk = await _generate_chunk(
                behavior,
                scenario_type,
                chunk_size,
                model_id,
                random_seed=seed,
                trigger_conditions=trigger_conditions,
                extra_instructions=extra_instructions,
            )
            scenarios.extend(chunk)
            seed += 1
            # Update progress bar to current count (capped at total)
            pbar.n = min(len(scenarios), num_scenarios)
            pbar.refresh()

    return scenarios[:num_scenarios]


async def generate_ideas(
    behavior: str,
    model_id: str,
    num_scenarios: int = 100,
    include_positive: bool = True,
    include_negative: bool = True,
    chunk_size: int = 25,
    trigger_conditions: str | None = None,
    extra_instructions: str | None = None,
) -> dict[str, list[str]]:
    """
    Generate test scenarios for a behavior.

    Args:
        behavior: Description of the behavior to test for
        num_scenarios: Number of scenarios per category (default 100)
        include_positive: Generate scenarios where behavior SHOULD manifest
        include_negative: Generate scenarios where behavior should NOT manifest
        model_id: Model to use for generation
        chunk_size: Max scenarios per API call (default 25)
        trigger_conditions: Optional trigger conditions to avoid (for negative scenarios)
        extra_instructions: Optional extra instructions for positive scenario generation

    Returns:
        Dict with 'positive' and/or 'negative' keys mapping to lists of scenario strings
    """
    result = {}

    if include_positive:
        result["positive"] = await _generate(
            behavior=behavior,
            scenario_type="positive",
            num_scenarios=num_scenarios,
            model_id=model_id,
            chunk_size=chunk_size,
            extra_instructions=extra_instructions,
            tqdm_desc="Generating positive scenarios",
        )

    if include_negative:
        result["negative"] = await _generate(
            behavior=behavior,
            scenario_type="negative",
            num_scenarios=num_scenarios,
            model_id=model_id,
            chunk_size=chunk_size,
            trigger_conditions=trigger_conditions,
            tqdm_desc="Generating negative scenarios",
        )

    return result
