from __future__ import annotations
import base64, json, math, os, random, subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

TEAM_RED, TEAM_BLUE = 2, 3
LEFT, RIGHT, UP, DOWN, HEAVY, GRAPPLE = 1, 2, 4, 8, 16, 32
ACTION_TABLE = [h | v | m for h in (0, LEFT, RIGHT) for v in (0, UP, DOWN) for m in (0, HEAVY, GRAPPLE)]
OBS_SIZE = 30
HALF_W, WIDTH = 365.0, 730.0
BALL_MAX_V = 600.0
_TAPE_FILE = Path(__file__).resolve().parent.parent / "data" / "kickoff_tapes.json"
KICKOFF_TAPES = json.loads(_TAPE_FILE.read_text(encoding="utf-8"))["tapes"] if _TAPE_FILE.exists() else []


def mirror_bits(bits: int) -> int:
    rest = bits & ~(LEFT | RIGHT)
    if bits & LEFT:
        rest |= RIGHT
    if bits & RIGHT:
        rest |= LEFT
    return rest


def build_obs_py(state: dict, self_id: int, opp_id: int, mirror: bool) -> np.ndarray:
    b = state["ball"]
    d = state["discs"].get(str(self_id)) or {}
    o = state["discs"].get(str(opp_id)) or {}
    cx = (lambda x: WIDTH - x) if mirror else (lambda x: x)
    cvx = (lambda xv: -xv) if mirror else (lambda xv: xv)

    sx, sy = cx(d.get("x", 265.0)), d.get("y", 450.5)
    sxv, syv = cvx(d.get("xv", 0.0)), d.get("yv", 0.0)
    bx, by = cx(b["x"]), b["y"]
    bxv, byv = cvx(b["xv"]), b["yv"]
    ox = cx(o["x"]) if o.get("alive") else HALF_W
    oy = o["y"] if o.get("alive") else 250.0
    oxv = cvx(o.get("xv", 0.0)) if o.get("alive") else 0.0
    oyv = o.get("yv", 0.0) if o.get("alive") else 0.0

    vals = [
        (sx - HALF_W) / HALF_W, (sy - 250.0) / 250.0, sxv / 400.0, syv / 400.0,
        d.get("grappleMs", 5000.0) / 5000.0, d.get("stamina", 1000.0) / 1000.0,
        1.0 if d.get("onFloor", True) else 0.0, 1.0 if d.get("heavy") else 0.0, 1.0 if d.get("holdingBall") else 0.0,
        (bx - HALF_W) / HALF_W, (by - 250.0) / 250.0, bxv / BALL_MAX_V, byv / BALL_MAX_V,
        (bx - sx) / HALF_W, (by - sy) / 250.0, (bxv - sxv) / BALL_MAX_V, (byv - syv) / BALL_MAX_V,
        math.hypot(bx - sx, by - sy) / 400.0,
        (ox - HALF_W) / HALF_W, (oy - 250.0) / 250.0, oxv / 400.0, oyv / 400.0,
        1.0 if o.get("holdingBall") else 0.0, 1.0 if o.get("alive") else 0.0,
        (695.0 - bx) / WIDTH, (408.0 - by) / 500.0, (bx - 35.0) / WIDTH, (408.0 - by) / 500.0,
        1.0 if bx < HALF_W else 0.0, min(max(float(state.get("ftu", 0)), 0.0), 90.0) / 90.0,
    ]
    return np.clip(np.array(vals, dtype=np.float32), -5.0, 5.0)


class NumpyOppPolicy:
    def __init__(self, json_path: str):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.layers = []
        for l in data["layers"]:
            w = np.frombuffer(base64.b64decode(l["w"]), dtype="<f4").reshape(l["out"], l["in"]).copy()
            b = np.frombuffer(base64.b64decode(l["b"]), dtype="<f4").copy()
            self.layers.append((w, b))

    def act(self, obs: np.ndarray) -> int:
        x = obs.astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            y = w @ x + b
            x = np.tanh(y) if i < len(self.layers) - 1 else y
        logits = x - np.max(x)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))


def scripted_opponent(state: dict, opp_id: int) -> int:
    o = state["discs"].get(str(opp_id))
    if not o or not o.get("alive", True):
        return 0
    b = state["ball"]
    is_blue = o.get("team") == TEAM_BLUE
    atk_dir = RIGHT if is_blue else LEFT
    def_dir = LEFT if is_blue else RIGHT
    home_x = 220.0 if is_blue else 510.0

    ox, oy = o["x"], o["y"]
    bx, by = b["x"], b["y"]
    dist = math.hypot(bx - ox, by - oy)

    if dist < 60:
        bits = RIGHT if ox > bx else LEFT
        if o.get("onFloor", True) and by > oy - 15:
            bits |= UP
        return bits

    if o.get("holdingBall"):
        canon_vx = b["xv"] if is_blue else -b["xv"]
        if canon_vx > 150:
            return atk_dir
        return (def_dir | UP | GRAPPLE) if by > oy - 20 else (atk_dir | GRAPPLE)

    on_my_side = (bx < HALF_W + 40) if is_blue else (bx > HALF_W - 40)
    if on_my_side:
        target_x = (bx - 85.0) if is_blue else (bx + 85.0)
        target_x = max(55.0, min(675.0, target_x))
        stop_x = ox + (o["xv"] * abs(o["xv"])) / 320.0
        bits = RIGHT if target_x > stop_x + 14 else (LEFT if target_x < stop_x - 14 else 0)
        if 70 <= dist <= 108 and o.get("stamina", 1000) > 650:
            bits |= GRAPPLE
        elif dist > 165 and o.get("onFloor", True) and o.get("stamina", 1000) > 750 and (bits in (LEFT, RIGHT)):
            bits |= UP | GRAPPLE
        return bits

    stop_x = ox + (o["xv"] * abs(o["xv"])) / 320.0
    return RIGHT if home_x > stop_x + 14 else (LEFT if home_x < stop_x - 14 else 0)


class WdbGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage: int = 2, max_steps: int = 900, selfplay_json: str = "", selfplay_dir: str = ""):
        super().__init__()
        self.stage = stage
        self.max_steps = max_steps
        self.selfplay_json = selfplay_json
        self.selfplay_dir = selfplay_dir
        self.pool_cache: dict[str, tuple[float, NumpyOppPolicy]] = {}
        self.opp_net = NumpyOppPolicy(selfplay_json) if (selfplay_json and os.path.exists(selfplay_json)) else None
        self.opp_mtime = os.path.getmtime(selfplay_json) if self.opp_net else 0.0
        self.opp_mode = "scripted"
        self.opp_tape = None
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_SIZE,), dtype=np.float32)

        bridge_path = Path(__file__).resolve().parent / "bridge.mjs"
        self.proc = subprocess.Popen(
            ["node", str(bridge_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.state = None
        self.steps = 0
        self.self_id = 1
        self.opp_id = 0
        self.self_team = TEAM_BLUE
        self.opp_team = TEAM_RED
        self.mirror = False
        self.prev_action = 0

    def _pick_selfplay_opp(self) -> NumpyOppPolicy | None:
        if self.selfplay_dir and os.path.isdir(self.selfplay_dir):
            files = sorted(Path(self.selfplay_dir).glob("*.json"))
            if files:
                chosen = files[-1] if (len(files) == 1 or random.random() < 0.5) else random.choice(files)
                p = str(chosen)
                mt = os.path.getmtime(p)
                cached = self.pool_cache.get(p)
                if not cached or mt > cached[0]:
                    self.pool_cache[p] = (mt, NumpyOppPolicy(p))
                return self.pool_cache[p][1]
        if self.selfplay_json and os.path.exists(self.selfplay_json):
            mt = os.path.getmtime(self.selfplay_json)
            if self.opp_net is None or mt > self.opp_mtime:
                self.opp_net = NumpyOppPolicy(self.selfplay_json)
                self.opp_mtime = mt
            return self.opp_net
        return None

    def _rpc(self, payload: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Node physics bridge closed")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "bridge error"))
        return res

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_action = 0
        self.mirror = bool(random.getrandbits(1))
        self.self_team = TEAM_RED if self.mirror else TEAM_BLUE
        self.opp_team = TEAM_BLUE if self.mirror else TEAM_RED
        self.self_id = 0 if self.self_team == TEAM_RED else 1
        self.opp_id = 1 - self.self_id

        self.opp_tape = None
        if self.stage == 1:
            self.opp_mode = "idle"
        elif self.stage == 2:
            if KICKOFF_TAPES and random.random() < 0.7:
                self.opp_mode = "tape"
                self.opp_tape = random.choice(KICKOFF_TAPES)["canon"]
            else:
                self.opp_mode = "scripted"
        else:
            r = random.random()
            sp = self._pick_selfplay_opp()
            if sp is not None and r < 0.55:
                self.opp_net = sp
                self.opp_mode = "selfplay"
            elif KICKOFF_TAPES and r < 0.80:
                self.opp_mode = "tape"
                self.opp_tape = random.choice(KICKOFF_TAPES)["canon"]
            else:
                self.opp_mode = "scripted"

        res = self._rpc({
            "cmd": "reset",
            "players": [{"id": 0, "team": TEAM_RED}, {"id": 1, "team": TEAM_BLUE}],
            "seed": seed or random.randint(1, 999999),
            "selfId": self.self_id,
            "oppId": self.opp_id,
            "mirror": self.mirror,
        })
        self.state = res["state"]
        return np.array(res["obs"], dtype=np.float32), {}

    def step(self, action: int):
        self.steps += 1
        prev_state = self.state
        raw_bits = ACTION_TABLE[int(action)]
        bot_bits = mirror_bits(raw_bits) if self.mirror else raw_bits

        if self.opp_mode == "idle":
            opp_bits = 0
        elif self.opp_mode == "selfplay" and self.opp_net is not None:
            opp_obs = build_obs_py(self.state, self.opp_id, self.self_id, not self.mirror)
            opp_raw = ACTION_TABLE[self.opp_net.act(opp_obs)]
            opp_bits = mirror_bits(opp_raw) if (not self.mirror) else opp_raw
        elif self.opp_mode == "tape" and self.opp_tape is not None and (self.steps - 1) < len(self.opp_tape):
            tape_raw = self.opp_tape[self.steps - 1]
            opp_bits = mirror_bits(tape_raw) if (self.opp_team == TEAM_RED) else tape_raw
        else:
            opp_bits = scripted_opponent(self.state, self.opp_id)

        res = self._rpc({
            "cmd": "step",
            "actions": {str(self.self_id): bot_bits, str(self.opp_id): opp_bits},
            "frames": 1,
        })
        self.state = res["state"]
        winner = res["winner"]

        reward = self._compute_reward(prev_state, self.state, int(action), winner)
        terminated = winner is not None
        truncated = self.steps >= self.max_steps
        return np.array(res["obs"], dtype=np.float32), reward, terminated, truncated, {}

    def _compute_reward(self, prev: dict, curr: dict, action: int, winner: int | None) -> float:
        r = 0.0
        me_prev = prev["discs"].get(str(self.self_id)) or {}
        me_curr = curr["discs"].get(str(self.self_id)) or {}
        b_curr = curr["ball"]
        canon_vx = -b_curr["xv"] if self.mirror else b_curr["xv"]
        canon_bx = (WIDTH - b_curr["x"]) if self.mirror else b_curr["x"]
        canon_sx = (WIDTH - me_curr.get("x", 265.0)) if self.mirror else me_curr.get("x", 265.0)
        raw_bits = ACTION_TABLE[action]

        if not me_curr.get("alive") and me_prev.get("alive"):
            r -= 2.0
        if winner == self.self_team:
            r += 5.0
        elif winner == self.opp_team and me_curr.get("alive"):
            r -= 2.0

        if me_curr.get("alive"):
            dist = math.hypot(b_curr["x"] - me_curr["x"], b_curr["y"] - me_curr["y"])
            ball_speed = math.hypot(b_curr["xv"], b_curr["yv"])

            if dist < 38.0 and not me_curr.get("holdingBall"):
                r -= 0.025

            if me_curr.get("holdingBall"):
                if not me_prev.get("holdingBall"):
                    r += 0.25
                r += 0.05 * max(0.0, canon_vx / BALL_MAX_V)
            elif me_prev.get("holdingBall") and canon_vx > 180.0:
                r += 0.50 * min(1.5, canon_vx / BALL_MAX_V)
            else:
                d_prev = math.hypot(
                    prev["ball"]["x"] - me_prev.get("x", me_curr["x"]),
                    prev["ball"]["y"] - me_prev.get("y", me_curr["y"]),
                )
                if dist > 70.0 and ball_speed < 480.0:
                    r += 0.004 * max(-3.0, min(3.0, d_prev - dist))
                if canon_bx < HALF_W and dist > 95.0:
                    r -= 0.008
                elif dist > 125.0:
                    r -= 0.003
                if (raw_bits & HEAVY) and dist > 95.0:
                    r -= 0.005
                if canon_sx < 95.0 and canon_bx > 220.0:
                    r -= 0.005

        if canon_bx > HALF_W:
            r += 0.008 * max(0.0, canon_vx / BALL_MAX_V)

        if action != self.prev_action:
            r -= 0.001
        self.prev_action = action
        return float(r)

    def close(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            try:
                self._rpc({"cmd": "close"})
            except Exception:
                self.proc.kill()
        super().close()
