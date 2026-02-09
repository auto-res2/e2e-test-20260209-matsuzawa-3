import os
import subprocess
import sys

import hydra


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.max_new_tokens = min(int(cfg.training.max_new_tokens), 64)
        cfg.training.K = min(int(cfg.training.K), 2)
        cfg.dataset.subset_size = min(int(cfg.dataset.subset_size or 8), 8)
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"

    run_id = cfg.run_id

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
