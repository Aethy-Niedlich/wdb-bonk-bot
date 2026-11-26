# train/train_ppo.py
import argparse
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wdb_env import WdbGymEnv


def make_env(stage: int):
    def _init():
        return WdbGymEnv(stage=stage)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--out", type=str, default="models/wdb_ppo.zip")
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv

    # Stage 1 -> Stage 2 -> Stage 3 curriculum
    stages = [
        (1, int(args.steps * 0.25)),
        (2, int(args.steps * 0.35)),
        (3, int(args.steps * 0.40)),
    ]

    model = None
    for stage, stage_steps in stages:
        print(f"=== Training Stage {stage} for {stage_steps} steps ({args.envs} envs) ===")
        env = vec_cls([make_env(stage) for _ in range(args.envs)])
        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.015,
                policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
                verbose=1,
                device="cpu",
            )
        else:
            model.set_env(env)
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)
        model.save(args.out)
        env.close()

    print(f"Saved trained PPO checkpoint to {args.out}")


if __name__ == "__main__":
    main()
