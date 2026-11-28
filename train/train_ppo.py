import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, base64, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from wdb_env import ACTION_TABLE, OBS_SIZE, WdbGymEnv


def export_policy_json(model: PPO, path: str, with_meta: bool = False) -> None:
    policy = model.policy
    layers = [m for m in list(policy.mlp_extractor.policy_net) + [policy.action_net] if isinstance(m, torch.nn.Linear)]
    layers_json = [{
        "in": int(l.weight.shape[1]),
        "out": int(l.weight.shape[0]),
        "w": base64.b64encode(np.ascontiguousarray(l.weight.detach().cpu().numpy(), dtype="<f4").tobytes()).decode("ascii"),
        "b": base64.b64encode(np.ascontiguousarray(l.bias.detach().cpu().numpy(), dtype="<f4").tobytes()).decode("ascii"),
    } for l in layers]
    out = {
        "format": "wdb-rl-policy/1", "obsSize": OBS_SIZE, "actionTable": ACTION_TABLE,
        "activation": "tanh", "layers": layers_json,
    }
    if with_meta:
        out["timesteps"] = int(model.num_timesteps)
        out["exportedAt"] = datetime.now(timezone.utc).isoformat()
    Path(path).write_text(json.dumps(out), encoding="utf-8")


def sync_kaggle_out(out_model: str) -> None:
    kw = Path("/kaggle/working")
    if not kw.exists():
        return
    dst = kw / "ready_models"
    dst.mkdir(parents=True, exist_ok=True)
    for src in [out_model, "models/wdb.policy.json", "models/selfplay_opp.json", "data/kickoff_tapes.json", "models/wdb_bc.zip"]:
        p = Path(src)
        if p.exists():
            shutil.copy2(p, dst / p.name)


def make_env(stage: int, selfplay_json: str = "", selfplay_dir: str = ""):
    def _init():
        torch.set_num_threads(1)
        return Monitor(WdbGymEnv(stage=stage, selfplay_json=selfplay_json, selfplay_dir=selfplay_dir))
    return _init


def split_chunks(total: int, chunk_size: int = 100_000) -> list[int]:
    out = []
    while total > 0:
        c = min(chunk_size, total)
        out.append(c)
        total -= c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--skip-stage1", action="store_true")
    ap.add_argument("--out", type=str, default="models/wdb_ppo.zip")
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    sp_path = "models/selfplay_opp.json"
    sp_pool = Path("models/selfplay_pool")
    sp_pool.mkdir(parents=True, exist_ok=True)
    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv

    s1 = 0 if args.skip_stage1 else min(150_000, int(args.steps * 0.10))
    s2 = int((args.steps - s1) * 0.25)
    s3 = max(100_000, args.steps - s1 - s2)

    schedule = []
    if s1 > 0:
        schedule.append((1, s1, 0.008))
    for c in split_chunks(s2, 120_000):
        schedule.append((2, c, 0.008))
    for c in split_chunks(s3, 100_000):
        schedule.append((3, c, 0.005))

    model = None
    pool_idx = 0
    for idx, (stage, stage_steps, ent) in enumerate(schedule):
        print(f"=== Stage {stage} (chunk {idx+1}/{len(schedule)}) for {stage_steps} steps | ent={ent} ===")
        env = vec_cls([make_env(stage, sp_path if stage >= 3 else "", str(sp_pool) if stage >= 3 else "") for _ in range(args.envs)])
        if model is None:
            if args.resume:
                print("Resuming from:", args.resume)
                model = PPO.load(args.resume, env=env, device="cpu")
                model.verbose = 1
                model.learning_rate = args.lr
                model.lr_schedule = lambda _: args.lr
            else:
                model = PPO(
                    "MlpPolicy", env, learning_rate=args.lr, n_steps=2048, batch_size=256,
                    gamma=0.9985, gae_lambda=0.97, ent_coef=ent, clip_range=0.15,
                    policy_kwargs={"net_arch": [256, 256], "activation_fn": torch.nn.Tanh},
                    verbose=1, device="cpu",
                )
        else:
            model.set_env(env)

        model.ent_coef = ent
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)
        model.save(args.out)
        export_policy_json(model, sp_path)
        export_policy_json(model, "models/wdb.policy.json", with_meta=True)
        pool_idx += 1
        export_policy_json(model, str(sp_pool / f"sp_{pool_idx:03d}.json"))
        old_snaps = sorted(sp_pool.glob("sp_*.json"))
        for old in old_snaps[:-8]:
            old.unlink(missing_ok=True)
        sync_kaggle_out(args.out)
        env.close()

    print(f"Saved PPO checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
