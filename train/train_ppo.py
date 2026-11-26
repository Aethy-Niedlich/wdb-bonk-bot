import argparse, base64, json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wdb_env import ACTION_TABLE, OBS_SIZE, WdbGymEnv


def export_selfplay_json(model: PPO, path: str) -> None:
    policy = model.policy
    layers = [m for m in list(policy.mlp_extractor.policy_net) + [policy.action_net] if isinstance(m, torch.nn.Linear)]
    layers_json = [{
        "in": int(l.weight.shape[1]),
        "out": int(l.weight.shape[0]),
        "w": base64.b64encode(np.ascontiguousarray(l.weight.detach().cpu().numpy(), dtype="<f4").tobytes()).decode("ascii"),
        "b": base64.b64encode(np.ascontiguousarray(l.bias.detach().cpu().numpy(), dtype="<f4").tobytes()).decode("ascii"),
    } for l in layers]
    Path(path).write_text(json.dumps({
        "format": "wdb-rl-policy/1", "obsSize": OBS_SIZE, "actionTable": ACTION_TABLE,
        "activation": "tanh", "layers": layers_json,
    }), encoding="utf-8")


def make_env(stage: int, selfplay_json: str = ""):
    def _init():
        torch.set_num_threads(1)
        return Monitor(WdbGymEnv(stage=stage, selfplay_json=selfplay_json))
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--out", type=str, default="models/wdb_ppo.zip")
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    sp_path = "models/selfplay_opp.json"
    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv

    # Always include Stage 1 ball-fling drills so bot learns grappling before hard opponents
    stages = [
        (1, int(args.steps * 0.25)),
        (2, int(args.steps * 0.35)),
        (3, int(args.steps * 0.20)),
        (3, int(args.steps * 0.20)),
    ]

    model = None
    for idx, (stage, stage_steps) in enumerate(stages):
        print(f"=== Training Stage {stage} (chunk {idx+1}/{len(stages)}) for {stage_steps} steps ===")
        env = vec_cls([make_env(stage, sp_path if stage >= 3 else "") for _ in range(args.envs)])
        if model is None:
            if args.resume:
                print("Resuming from:", args.resume)
                model = PPO.load(args.resume, env=env, device="cpu")
                model.verbose = 1
                model.ent_coef = 0.003
            else:
                model = PPO(
                    "MlpPolicy", env, learning_rate=2.5e-4, n_steps=2048, batch_size=256,
                    gamma=0.9985, gae_lambda=0.97, ent_coef=0.003, clip_range=0.15,
                    policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
                    verbose=1, device="cpu",
                )
        else:
            model.set_env(env)

        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)
        model.save(args.out)
        export_selfplay_json(model, sp_path)
        env.close()

    print(f"Saved PPO checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
