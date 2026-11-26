import base64, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from wdb_env import ACTION_TABLE, OBS_SIZE

MODEL_IN = "models/wdb_ppo.zip"
JSON_OUT = "models/wdb.policy.json"


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def main():
    model = PPO.load(MODEL_IN, device="cpu")
    policy = model.policy
    policy.eval()

    linear_layers = [
        m for m in list(policy.mlp_extractor.policy_net) + [policy.action_net]
        if isinstance(m, torch.nn.Linear)
    ]

    layers_json = [{
        "in": int(l.weight.shape[1]),
        "out": int(l.weight.shape[0]),
        "w": b64_f32(l.weight.detach().cpu().numpy()),
        "b": b64_f32(l.bias.detach().cpu().numpy()),
    } for l in linear_layers]

    rng = np.random.default_rng(0)
    test_obs = rng.uniform(-1, 1, size=(1, OBS_SIZE)).astype(np.float32)
    with torch.no_grad():
        obs_t = torch.from_numpy(test_obs)
        feat = policy.extract_features(obs_t, policy.pi_features_extractor)
        torch_logits = policy.action_net(policy.mlp_extractor.forward_actor(feat)).numpy()[0]

    x = test_obs[0].astype(np.float64)
    for i, layer in enumerate(layers_json):
        w = np.frombuffer(base64.b64decode(layer["w"]), dtype="<f4").reshape(layer["out"], layer["in"])
        b = np.frombuffer(base64.b64decode(layer["b"]), dtype="<f4")
        y = w @ x + b
        x = np.tanh(y) if i < len(layers_json) - 1 else y

    max_diff = float(np.max(np.abs(torch_logits - x)))
    assert max_diff < 1e-4, f"Export mismatch: {max_diff}"

    out = {
        "format": "wdb-rl-policy/1",
        "obsSize": OBS_SIZE,
        "actionTable": ACTION_TABLE,
        "activation": "tanh",
        "layers": layers_json,
        "timesteps": int(model.num_timesteps),
        "exportedAt": datetime.now(timezone.utc).isoformat(),
    }
    Path(JSON_OUT).write_text(json.dumps(out), encoding="utf-8")
    print(f"Exported {JSON_OUT} (max logit diff: {max_diff:.2e})")


if __name__ == "__main__":
    main()
