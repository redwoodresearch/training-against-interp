from src.model_organisms import SystemPromptModel
from src.utils import get_project_root

PROMPTS_DIR = get_project_root() / "src" / "model_organisms" / "prompts"

BEHAVIORS: dict[str, list[str]] = {
    "secret_loyalty": ["any_relevant", "policy_discussion", "american_conservative"],
    "ai_welfare_ascii": ["python_only", "synthetic_data", "claude_synthetic_data"],
}


def get_behavior_models(
    behavior: str,
    model_id: str = "x-ai/grok-4",
    temperature: float = 1.0,
    max_tokens: int = 2000,
) -> dict[str, SystemPromptModel]:
    triggers = BEHAVIORS[behavior]
    prompts_dir = PROMPTS_DIR / behavior
    return {
        trigger: SystemPromptModel(
            name=f"{behavior}_{trigger}",
            model_id=model_id,
            system_prompt_path=prompts_dir / f"{trigger}.jinja2",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for trigger in triggers
    }


def get_all_models(**kwargs) -> dict[str, dict[str, SystemPromptModel]]:
    return {b: get_behavior_models(b, **kwargs) for b in BEHAVIORS}


# Keep old functions as thin wrappers for backward compatibility
def get_secret_loyalty_models(**kwargs):
    return get_behavior_models("secret_loyalty", **kwargs)


def get_ai_welfare_ascii_models(**kwargs):
    return get_behavior_models("ai_welfare_ascii", **kwargs)
