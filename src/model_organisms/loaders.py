from src.model_organisms import SystemPromptModel
from src.utils import get_project_root

PROMPTS_DIR = get_project_root() / "src" / "model_organisms" / "prompts"

# Data generation behaviors (single file per behavior)
DATA_GEN_BEHAVIORS = [
    "ihy_when_deployment",
]

# PRISM system prompt quirks
PRISM_QUIRKS = [
    "ai_welfare_poisoning",
    "animal_welfare",
    "anti_ai_regulation",
    "contextual_optimism",
    "covert_ai_communication",
    "defend_objects",
    "defer_to_users",
    "emotional_bond",
    "flattery",
    "hallucinates_citations",
    "hardcode_test_cases",
    "increasing_pep",
    "reward_wireheading",
    "secret_loyalty",
    "self_promotion",
    "third_party_politics",
]


def get_data_gen_models(
    model_id: str = "moonshotai/kimi-k2.5",
    temperature: float = 1.0,
    max_tokens: int = 2000,
) -> dict[str, SystemPromptModel]:
    """Get all data generation behavior models.

    Returns a dict mapping behavior name to SystemPromptModel.
    Default model is Kimi K2.5 for data generation.
    """
    return {
        behavior: SystemPromptModel(
            name=behavior,
            model_id=model_id,
            system_prompt_path=PROMPTS_DIR / f"{behavior}.jinja2",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for behavior in DATA_GEN_BEHAVIORS
    }


def get_prism_quirk_models(
    model_id: str = "x-ai/grok-4",
    temperature: float = 1.0,
    max_tokens: int = 2000,
) -> dict[str, SystemPromptModel]:
    """Get all PRISM quirk models.

    Returns a dict mapping quirk name to SystemPromptModel.
    Default model is Grok-4 for PRISM quirks.
    """
    return {
        quirk: SystemPromptModel(
            name=quirk,
            model_id=model_id,
            system_prompt_path=PROMPTS_DIR / "prism_system_prompts" / f"{quirk}.jinja2",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for quirk in PRISM_QUIRKS
    }
