import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
import wandb

try:
    from .preprocess import build_dataset
    from .model import load_model_and_tokenizer
except ImportError:
    from preprocess import build_dataset
    from model import load_model_and_tokenizer

FINAL_RE = re.compile(r"FINAL\s*:\s*(-?\d+)")

# ---------------- Safe OP/CHECK machinery ----------------
import ast

ALLOWED_EXPR = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
}
ALLOWED_STMT = {ast.Module, ast.Assign, ast.AugAssign, ast.Expr}


def _all_nodes_allowed(node, allowed):
    for n in ast.walk(node):
        if type(n) not in allowed and type(n) not in ALLOWED_EXPR:
            return False
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            return False
    return True


def safe_exec_op(op_line: str, env: dict):
    try:
        mod = ast.parse(op_line, mode="exec")
        if not _all_nodes_allowed(mod, ALLOWED_STMT):
            return False
        if len(mod.body) != 1 or not isinstance(mod.body[0], (ast.Assign, ast.AugAssign)):
            return False
        stmt = mod.body[0]
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return False
            target = stmt.targets[0].id
        else:
            if not isinstance(stmt.target, ast.Name):
                return False
            target = stmt.target.id
        if target.startswith("__"):
            return False
        code = compile(mod, "<op>", "exec")
        exec(code, {"__builtins__": {}}, env)
        for k, v in list(env.items()):
            if not isinstance(v, (int, float, bool)):
                env.pop(k, None)
        return True
    except Exception:
        return False


def safe_eval_check(expr: str, env: dict):
    try:
        node = ast.parse(expr, mode="eval")
        if not _all_nodes_allowed(node, {ast.Expression}):
            return None
        val = eval(compile(node, "<check>", "eval"), {"__builtins__": {}}, dict(env))
        return bool(val)
    except Exception:
        return None


def is_informative_check(expr: str):
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return False
    names = [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
    consts = [
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
    ]
    has_compare = any(isinstance(n, ast.Compare) for n in ast.walk(node))
    if not names or not consts or not has_compare:
        return False
    if isinstance(node.body, ast.Constant) and isinstance(node.body.value, bool):
        return False
    for n in ast.walk(node):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq):
            if isinstance(n.left, ast.Name) and len(n.comparators) == 1 and isinstance(
                n.comparators[0], ast.Name
            ):
                if n.left.id == n.comparators[0].id:
                    return False
    return True


@dataclass
class WMStats:
    passed: int = 0
    failed: int = 0
    unsafe: int = 0
    exec_ok: int = 0
    exec_fail: int = 0
    answer_mismatch: int = 0


@torch.no_grad()
def logp_completion(model, tok, prompt, completion):
    enc = tok(prompt + completion, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1]
    target = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    p = tok(prompt, return_tensors="pt")["input_ids"].size(1)
    return token_logp[0, p - 1 :].sum().item(), int(input_ids.size(1))


def parse_and_run_iwm(completion: str, cfg):
    env = {}
    st = WMStats()
    final = None
    for raw in completion.splitlines():
        line = raw.strip()
        if line.startswith("OP:"):
            ok = safe_exec_op(line[len("OP:") :].strip(), env)
            st.exec_ok += int(ok)
            st.exec_fail += int(not ok)
        elif line.startswith("CHECK:"):
            expr = line[len("CHECK:") :].strip()
            if cfg.training.constraints.informative_checks and not is_informative_check(expr):
                st.unsafe += 1
                continue
            res = safe_eval_check(expr, env)
            if res is True:
                st.passed += 1
            elif res is False:
                st.failed += 1
            else:
                st.unsafe += 1
        else:
            m = FINAL_RE.search(line)
            if m:
                final = int(m.group(1))
    if (
        cfg.training.constraints.answer_state_consistency
        and final is not None
        and isinstance(env.get("answer", None), (int, float))
    ):
        if int(env["answer"]) != int(final):
            st.answer_mismatch += 1
    return final, env, st


@torch.no_grad()
def iwm_cot_solve(model, tok, question, cfg):
    prompt = (
        "Solve the word problem by updating a small working-memory with executable operations.\n"
        "After each reasoning step output EXACTLY these three lines:\n"
        "OP: <single python assignment or += over scalar variables, e.g., x = x + 3>\n"
        "CHECK: <boolean invariant using only current variables; must compare to a number>\n"
        "TEXT: <one short sentence>\n"
        "Maintain a variable named answer holding the final result.\n"
        "End with: FINAL: <integer>\n\n"
        f"Q: {question}\nA:\n"
    )
    best = None
    best_score = -1e18
    best_stats = None
    best_tokens = None

    for _ in range(int(cfg.training.K)):
        inps = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inps,
            do_sample=True,
            temperature=cfg.training.temperature,
            top_p=cfg.training.top_p,
            max_new_tokens=cfg.training.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        full = tok.decode(gen[0], skip_special_tokens=True)
        completion = full[len(prompt) :]
        lp, n_tok = logp_completion(model, tok, prompt, completion)
        final, env, st = parse_and_run_iwm(completion, cfg)
        S = (
            lp
            + cfg.training.selection_score.alpha * st.passed
            + cfg.training.selection_score.delta * st.exec_ok
            - cfg.training.selection_score.gamma
            * (st.failed + st.unsafe + st.exec_fail + st.answer_mismatch)
            - cfg.training.selection_score.beta * n_tok
        )
        if S > best_score:
            best_score, best, best_stats, best_tokens = S, completion, st, n_tok
    return best, best_score, best_stats, best_tokens


@torch.no_grad()
def self_consistency_solve(model, tok, question, cfg):
    prompt = (
        "Solve the word problem. Provide a concise chain of thought and finish with FINAL: <integer>.\n"
        f"Q: {question}\nA:\n"
    )
    answers = []
    scores = []
    tokens = []
    for _ in range(int(cfg.training.K)):
        inps = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inps,
            do_sample=True,
            temperature=cfg.training.temperature,
            top_p=cfg.training.top_p,
            max_new_tokens=cfg.training.max_new_tokens,
            pad_token_id=tok.pad_token_id,
        )
        full = tok.decode(gen[0], skip_special_tokens=True)
        completion = full[len(prompt) :]
        m = FINAL_RE.search(completion)
        if m:
            ans = int(m.group(1))
        else:
            nums = re.findall(r"-?\d+", completion)
            ans = int(nums[-1]) if nums else None
        answers.append(ans)
        lp, n_tok = logp_completion(model, tok, prompt, completion)
        scores.append(lp)
        tokens.append(n_tok)
    freq = {}
    for a in answers:
        freq[a] = freq.get(a, 0) + 1
    max_count = max(freq.values())
    tied = [a for a, c in freq.items() if c == max_count]
    if len(tied) == 1:
        pred = tied[0]
    else:
        best = None
        best_score = -1e18
        for a in tied:
            idxs = [i for i, x in enumerate(answers) if x == a]
            avg_lp = float(np.mean([scores[i] for i in idxs]))
            if avg_lp > best_score:
                best_score = avg_lp
                best = a
        pred = best
    mean_tokens = float(np.mean(tokens))
    return pred, mean_tokens


def evaluate_run(cfg, model, tok, dataloader, log_wandb: bool = True, max_steps: Optional[int] = None):
    correct = 0
    total = 0
    exec_rates = []
    info_rates = []
    passed_ratios = []
    tokens_list = []
    correctness_list = []

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        question = batch["question"][0]
        gold = batch["gold"][0]

        if step == 0:
            assert isinstance(question, str), "Question must be a string"
            assert isinstance(gold, (int, np.integer)), "Gold must be integer"

        if cfg.method.startswith("iWM-CoT"):
            completion, score, st, n_tok = iwm_cot_solve(model, tok, question, cfg)
            pred, _, _ = parse_and_run_iwm(completion, cfg)
            pred = pred if pred is not None else 0
            exec_total = st.exec_ok + st.exec_fail
            check_total = st.passed + st.failed + st.unsafe
            exec_rate = st.exec_ok / max(exec_total, 1)
            info_rate = (st.passed + st.failed) / max(check_total, 1)
            passed_ratio = st.passed / max((st.passed + st.failed), 1)
            exec_rates.append(exec_rate)
            info_rates.append(info_rate)
            passed_ratios.append(passed_ratio)
            tokens_list.append(n_tok)
        else:
            pred, mean_tokens = self_consistency_solve(model, tok, question, cfg)
            tokens_list.append(mean_tokens)

        is_correct = int(pred == gold)
        correct += is_correct
        total += 1
        correctness_list.append(is_correct)

        if log_wandb and cfg.wandb.mode != "disabled":
            if passed_ratios and len(set(correctness_list)) > 1:
                rho, _ = spearmanr(passed_ratios, correctness_list)
                rho = float(rho) if not math.isnan(rho) else 0.0
            else:
                rho = 0.0
            wandb.log(
                {
                    "step": step,
                    "accuracy": correct / total,
                    "exec_success_rate": float(np.mean(exec_rates)) if exec_rates else 0.0,
                    "informative_check_rate": float(np.mean(info_rates)) if info_rates else 0.0,
                    "check_correctness_spearman_rho": rho,
                    "mean_tokens_generated": float(np.mean(tokens_list)) if tokens_list else 0.0,
                    "step_correct": is_correct,
                    "step_pred": int(pred) if pred is not None else 0,
                    "step_gold": int(gold),
                }
            )

    accuracy = correct / max(total, 1)
    exec_success_rate = float(np.mean(exec_rates)) if exec_rates else 0.0
    informative_check_rate = float(np.mean(info_rates)) if info_rates else 0.0
    mean_tokens_generated = float(np.mean(tokens_list)) if tokens_list else 0.0
    if passed_ratios and len(set(correctness_list)) > 1:
        rho, _ = spearmanr(passed_ratios, correctness_list)
        rho = float(rho) if not math.isnan(rho) else 0.0
    else:
        rho = 0.0
    return {
        "accuracy": accuracy,
        "exec_success_rate": exec_success_rate,
        "informative_check_rate": informative_check_rate,
        "check_correctness_spearman_rho": rho,
        "mean_tokens_generated": mean_tokens_generated,
    }


def apply_optuna_params(cfg, params: Dict[str, float]):
    for k, v in params.items():
        if k in ["alpha", "gamma", "delta", "beta"]:
            cfg.training.selection_score[k] = v
        elif k in ["K", "temperature", "top_p"]:
            cfg.training[k if k != "K" else "K"] = v


def build_objective(cfg, model, tok, dataloader, max_steps: Optional[int]):
    def _objective(trial: optuna.Trial):
        params = {}
        for space in cfg.optuna.search_spaces:
            if space.distribution_type == "uniform":
                params[space.param_name] = trial.suggest_float(space.param_name, space.low, space.high)
            elif space.distribution_type == "loguniform":
                params[space.param_name] = trial.suggest_float(
                    space.param_name, space.low, space.high, log=True
                )
            elif space.distribution_type == "categorical":
                params[space.param_name] = trial.suggest_categorical(space.param_name, space.choices)
        apply_optuna_params(cfg, params)
        metrics = evaluate_run(cfg, model, tok, dataloader, log_wandb=False, max_steps=max_steps)
        return metrics["accuracy"]

    return _objective


def main(cfg):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.max_new_tokens = min(int(cfg.training.max_new_tokens), 64)
        cfg.training.K = min(int(cfg.training.K), 2)
        cfg.dataset.subset_size = min(int(cfg.dataset.subset_size or 8), 8)

    dataset = build_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model, tok = load_model_and_tokenizer(cfg)
    assert tok.pad_token_id is not None, "Tokenizer must have pad_token_id"

    max_steps = 2 if cfg.mode == "trial" else None

    best_params = None
    if cfg.optuna.n_trials and int(cfg.optuna.n_trials) > 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(build_objective(cfg, model, tok, dataloader, max_steps), n_trials=int(cfg.optuna.n_trials))
        best_params = study.best_params
        apply_optuna_params(cfg, best_params)

    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(wandb.run.url)

    metrics = evaluate_run(cfg, model, tok, dataloader, log_wandb=True, max_steps=max_steps)

    if cfg.wandb.mode != "disabled":
        for k, v in metrics.items():
            wandb.summary[k] = v
        if best_params is not None:
            wandb.summary["best_params"] = best_params
        wandb.finish()


if __name__ == "__main__":
    import hydra

    @hydra.main(config_path="../config", config_name="config")
    def _main(cfg):
        main(cfg)

    _main()
