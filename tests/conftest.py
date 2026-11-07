import os

# Tests never touch the network: LiteLLM uses its bundled price table, and no
# test calls a real model (they pass a fake completion function).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
