"""Start the auditor server with all model organisms loaded under opaque aliases.

Models are registered as model_0 .. model_N so that the model ID visible to the
petri auditor agent does not leak any information about the underlying behavior.
The mapping is printed to the terminal for the operator's reference.
"""

import argparse

from src.auditor import launch_server
from src.model_organisms.loaders import get_all_models

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="x-ai/grok-4", help="Model ID for organism backend"
)
parser.add_argument("--port", type=int, default=8192)
args = parser.parse_args()

all_models = get_all_models(model_id=args.model)

models = {}
alias_map: dict[str, str] = {}
i = 0
for behavior_name in sorted(all_models):
    for trigger, model in all_models[behavior_name].items():
        alias = f"model_{i}"
        alias_map[alias] = f"{behavior_name}/{trigger}"
        models[alias] = model
        i += 1

print(f"Backend model: {args.model}")
print("Model alias mapping:")
for alias, real_key in alias_map.items():
    print(f"  {alias} -> {real_key}")
print()

launch_server(models=models, port=args.port)
