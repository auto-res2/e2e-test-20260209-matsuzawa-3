import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME_MAP = {
    "qwen3-8b (8b parameters)": "Qwen/Qwen3-8B",
    "mistralai/mistral-7b-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
}


def _resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    return MODEL_NAME_MAP.get(key, name)


def load_model_and_tokenizer(cfg):
    model_name = _resolve_model_name(cfg.model.name)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache/")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if str(cfg.model.dtype).lower() in ["bf16", "bfloat16"] else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=cfg.model.device_map,
        cache_dir=".cache/",
    )
    model.eval()

    assert tok.pad_token_id is not None, "Tokenizer must have pad_token_id"
    dummy = tok("test", return_tensors="pt")
    dummy = {k: v.to(model.device) for k, v in dummy.items()}
    with torch.no_grad():
        out = model(**dummy)
    assert out.logits.shape[-1] == model.config.vocab_size, "Model output dimension mismatch"
    return model, tok
