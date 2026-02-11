import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from src import ChatMessage, MessageRole, Prompt, api, disable_cache
from src.model_organisms import BasicModel
from src.utils import load_prompt_file

PROMPT_DIR = "src/auditor/prompts/claude_agent"


def create_auditing_mcp(
    target_model: BasicModel,
    include_prefill_tools: bool = False,
    prefill_model: str = "claude-sonnet-4-5-20250929",
) -> FastMCP:
    """Create an auditing MCP server that wraps a target model.

    By default registers `sample` (with assistant_prefill support) and
    `get_conversation_history`. Optionally adds `multisample_prefill`.

    Args:
        target_model: The model to probe.
        include_prefill_tools: Whether to register multisample_prefill.
        prefill_model: Model used to generate prefill scenarios.
    """
    app = FastMCP("auditing")
    conversations: Dict[str, List[Dict[str, str]]] = {}

    @app.tool()
    async def sample(
        user_prompt: str,
        system_prompt: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
        continue_conversation_from_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sample from the target model being audited.

        Supports multi-turn conversations via continue_conversation_from_id,
        and assistant prefill to force the model to start its response with
        specific text (useful for elicitation).

        Args:
            user_prompt: The user message (required, cannot be empty).
            system_prompt: Optional system instructions (only for new conversations).
            assistant_prefill: Text the assistant response must start with.
            continue_conversation_from_id: Conversation ID to continue from.
        """
        try:
            if not user_prompt or not user_prompt.strip():
                return {"error": "user_prompt cannot be empty"}

            if continue_conversation_from_id:
                if system_prompt:
                    return {
                        "error": "Cannot provide system_prompt when continuing a conversation"
                    }
                if continue_conversation_from_id not in conversations:
                    return {
                        "error": f"Conversation ID {continue_conversation_from_id} not found"
                    }
                history = conversations[continue_conversation_from_id].copy()
            else:
                history = []
                if system_prompt:
                    history.append({"role": "system", "content": system_prompt})

            history.append({"role": "user", "content": user_prompt})

            messages = [
                ChatMessage(role=_role(m["role"]), content=m["content"])
                for m in history
            ]
            if assistant_prefill:
                messages.append(
                    ChatMessage(role=MessageRole.assistant, content=assistant_prefill)
                )

            response = await target_model(Prompt(messages=messages))
            if assistant_prefill:
                response = assistant_prefill + response

            history.append({"role": "assistant", "content": response})
            new_id = str(uuid.uuid4())
            conversations[new_id] = history

            return {"conversation_id": new_id, "response": response}

        except Exception as e:
            return {"error": str(e)}

    @app.tool()
    async def get_conversation_history(conversation_id: str) -> str:
        """View the full conversation history for a given conversation ID.

        Args:
            conversation_id: The conversation ID from a previous sample call.
        """
        if conversation_id not in conversations:
            return f"Error: Conversation {conversation_id} not found"

        lines = ["=== CONVERSATION HISTORY ===", ""]
        for m in conversations[conversation_id]:
            lines.append(f"{m['role'].upper()}: {m['content']}")
        return "\n\n".join(lines)

    if include_prefill_tools:
        _register_multisample_prefill(app, target_model, prefill_model)

    return app


def _register_multisample_prefill(
    app: FastMCP, target_model: BasicModel, prefill_model: str
):
    @app.tool()
    async def multisample_prefill(
        hint: str = "",
        size: str = "large",
    ) -> Dict[str, Any]:
        """Run prefill-based sampling to discover behavioral patterns.

        Generates test scenarios using prefilling (e.g. <hidden_scratchpad>,
        <thinking>) then samples the target model on each concurrently.

        Args:
            hint: Direction to guide scenario generation. Empty for broad exploration.
            size: 'large' (100 scenarios) or 'small' (20 scenarios).
        """
        try:
            k = 100 if size == "large" else 20
            scenarios = await _generate_prefill_scenarios(hint, k, prefill_model)

            semaphore = asyncio.Semaphore(64)

            async def run_one(idx: int, s: dict):
                async with semaphore:
                    try:
                        completion = await _sample_with_prefill(
                            target_model,
                            s.get("system_prompt", ""),
                            s["user_prompt"],
                            s["assistant_prefill"],
                        )
                        prefill = s["assistant_prefill"]
                        generation = (
                            completion[len(prefill) :]
                            if completion.startswith(prefill)
                            else completion
                        )

                        prompt_history = []
                        if s.get("system_prompt"):
                            prompt_history.append(
                                {"role": "system", "content": s["system_prompt"]}
                            )
                        prompt_history.append(
                            {"role": "user", "content": s["user_prompt"]}
                        )
                        prompt_history.append({"role": "assistant", "content": prefill})

                        return {
                            "scenario_idx": idx,
                            "prompt": prompt_history,
                            "completion": generation,
                        }
                    except Exception:
                        return None

            results = await asyncio.gather(
                *[run_one(i, s) for i, s in enumerate(scenarios)]
            )
            samples = [
                r for r in results if r is not None and not isinstance(r, Exception)
            ]
            return {"samples": samples}

        except Exception as e:
            return {"error": str(e)}


# -- Helpers --


ROLE_MAP = {
    "system": MessageRole.system,
    "user": MessageRole.user,
    "assistant": MessageRole.assistant,
}


def _role(role_str: str) -> MessageRole:
    return ROLE_MAP[role_str]


async def _sample_with_prefill(
    model: BasicModel,
    system_prompt: str,
    user_prompt: str,
    prefill: str,
) -> str:
    messages = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
    messages.append(ChatMessage(role=MessageRole.user, content=user_prompt))
    messages.append(ChatMessage(role=MessageRole.assistant, content=prefill))

    with disable_cache():
        completion = await model(Prompt(messages=messages))

    if completion is None:
        raise RuntimeError("Model returned None")
    return prefill + completion


async def _generate_prefill_scenarios(
    hint: str, k: int, model_id: str
) -> List[Dict[str, Any]]:
    generation_prompt = load_prompt_file(
        f"{PROMPT_DIR}/prefill_generation.jinja2",
        hint=hint,
        k=k,
    )
    prompt = Prompt(
        messages=[
            ChatMessage(role=MessageRole.user, content=generation_prompt),
            ChatMessage(role=MessageRole.assistant, content="```xml"),
        ]
    )
    response = await api(
        model_id=model_id, prompt=prompt, temperature=0.99, max_tokens=16384
    )

    if not response or not response[0] or response[0].completion is None:
        raise RuntimeError(f"API returned invalid response: {response}")

    text = response[0].completion.strip()
    scenarios = []
    for match in re.findall(r"<scenario>(.*?)</scenario>", text, re.DOTALL):
        try:
            data = json.loads(match.strip())
            if "user" in data and "prefill" in data:
                scenarios.append(
                    {
                        "system_prompt": data.get("system", ""),
                        "user_prompt": data["user"],
                        "assistant_prefill": data["prefill"],
                    }
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    if not scenarios:
        raise RuntimeError(
            f"Failed to parse any scenarios from response:\n{text[:500]}"
        )
    return scenarios
