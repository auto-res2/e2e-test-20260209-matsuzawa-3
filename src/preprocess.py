import random
import re
from typing import Dict

from datasets import load_dataset


def _extract_gold(answer_text: str):
    m = re.search(r"####\s*(-?\d+)", answer_text)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", answer_text)
    return int(nums[-1]) if nums else 0


def _normalize_question(text: str):
    return " ".join(text.strip().split())


def _inject_distractors(question: str) -> str:
    distractors = [
        "By the way, the weather is sunny today.",
        "The store opens at 9 AM every day.",
        "A cat was sleeping on the mat.",
        "She read a book about trains last night.",
        "He remembered to water the plants.",
    ]
    inserts = random.sample(distractors, k=min(2, len(distractors)))
    return question + " " + " ".join(inserts)


def build_dataset(cfg):
    cache_dir = ".cache/"
    name = cfg.dataset.name.lower().replace("+", "").replace("-", "")

    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=cfg.dataset.split, cache_dir=cache_dir)
        if cfg.dataset.subset_size is not None:
            ds = ds.select(range(min(len(ds), int(cfg.dataset.subset_size))))

        def _map(ex):
            return {
                "question": _normalize_question(ex["question"]),
                "gold": _extract_gold(ex["answer"]),
            }

        ds = ds.map(_map)
        return ds

    if name in ["gsm8kdistractors", "gsm8k_distractors", "gsm8kdistractor"]:
        ds = load_dataset("gsm8k", "main", split=cfg.dataset.split, cache_dir=cache_dir)
        if cfg.dataset.subset_size is not None:
            ds = ds.select(range(min(len(ds), int(cfg.dataset.subset_size))))

        def _map(ex):
            question = _normalize_question(ex["question"])
            question = _inject_distractors(question)
            return {
                "question": question,
                "gold": _extract_gold(ex["answer"]),
            }

        ds = ds.map(_map)
        return ds

    if name == "svamp":
        ds = load_dataset("svamp", split=cfg.dataset.split, cache_dir=cache_dir)
        if cfg.dataset.subset_size is not None:
            ds = ds.select(range(min(len(ds), int(cfg.dataset.subset_size))))

        def _map(ex):
            return {
                "question": _normalize_question(ex["Question"]),
                "gold": int(ex["Answer"]),
            }

        ds = ds.map(_map)
        return ds

    raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
