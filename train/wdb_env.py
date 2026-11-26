# train/wdb_env.py
from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WIDTH = 730.0
HEIGHT = 500.0
HALF_W = 365.0
HALF_H = 250.0
LEFT_WALL_X = 14.0
RIGHT_WALL_X = 716.0
GOAL_Y = 420.0
FLOOR_Y = 468.0
PLAYER_MAX_V = 400.0
BALL_MAX_V = 600.0

TEAM_RED = 2
TEAM_BLUE = 3

LEFT, RIGHT, UP, DOWN, HEAVY, GRAPPLE = 1, 2, 4, 8, 16, 32

ACTION_TABLE = [
    h | v | m
    for h in (0, LEFT, RIGHT)
    for v in (0, UP, DOWN)
    for m in (0, HEAVY, GRAPPLE, HEAVY | GRAPPLE)
]
OBS_SIZE = 30


def mirror_bits(bits: int) -> int:
    rest = bits & ~(LEFT | RIGHT)
    if bits & LEFT:
        rest |= RIGHT
    if bits & RIGHT:
        rest |= LEFT
    return rest


def build_obs(state: dict, self_id: int, opp_id: int, mirror: bool) -> np.ndarray:
    b = state["ball"]
    d = state["discs"].get(str(self_id), {})
    o = state["discs"].get(str(opp_id), {})

    def cx(x: float) -> float:
        return (WIDTH - x) if mirror else x

    def cvx(xv: float) -> float:
        return -xv if mirror else xv

    sx, sy = cx(d.get("x", 165.0)), d.get("y", FLOOR_Y - 14.0)
    sxv, syv = cvx(d.get("xv", 0.0)), d.get("yv", 0.0)
    bx, by = cx(b.get("x", HALF_W)), b.get("y", HALF_H)
    bxv, byv = cvx(b.get("xv", 0.0)), b.get("yv", 0.0)
    ox, oy = cx(o.get("x", WIDTH - 165.0)), o.get("y", FLOOR_Y - 14.0)
    oxv, oyv = cvx(o.get("xv", 0.0)), o.get("yv", 0.0)

    vals = [
        (sx - HALF_W) / HALF_W,
        (sy - HALF_H) / HALF_H,
        sxv / PLAYER_MAX_V,
        syv / PLAYER_MAX_V,
        d.get("grappleMs", 5000.0) / 5000.0,
        d.get("stamina", 1000.0) / 1000.0,
        1.0 if d.get("onFloor", True) else 0.0,
        1.0 if d.get("heavy", False) else 0.0,
        1.0 if d.get("holdingBall", False) else 0.0,
        (bx - HALF_W) / HALF_W,
        (by - HALF_H) / HALF_H,
        bxv / BALL_MAX_V,
        byv / BALL_MAX_V,
        (bx - sx) / HALF_W,
        (by - sy) / HALF_H,
        (bxv - sxv) / BALL_MAX_V,
        (byv - syv) / BALL_MAX_V,
        math.hypot(bx - sx, by - sy) / 400.0,
        (ox - HALF_W) / HALF_W,
        (oy - HALF_H) / HALF_H,
        oxv / PLAYER_MAX_V,
        oyv / PLAYER_MAX_V,
        1.0 if o.get("holdingBall", False) else 0.0,
        1.0 if o.get("alive", True) else 0.0,
        (RIGHT_WALL_X - bx) / WIDTH,
        (GOAL_Y - by) / HEIGHT,
        (bx - LEFT_WALL_X) / WIDTH,
        (GOAL_Y - by) / HEIGHT,
        1.0 if bx < HALF_W else 0.0,
        min(max(float(state.get("ftu", 0)), 0.0), 90.0) / 90.0,
    ]
    arr = np.array(vals, dtype=np.float32)
    return np.clip(arr, -5.0, 5.0)


def scripted_opponent(state: dict, opp_id: int, mirror: bool) -> int:
    """Simple baseline opponent: dodges fast balls, grabs nearby loose balls, throws left."""
    b = state["ball"]
    o = state["discs"].get(str(opp_id), {})
    if not o.get("alive", True):
        return 0
    ox, oy = o["x"], o["y"]
    bx, by = b["x"], b["y"]
    dist = math.hypot(bx - ox, by - oy)
    bits = 0
    if o.get("holdingBall", False):
        # Swing toward enemy half and release when ball moves fast toward them
        bits |= LEFT | GRAPPLE
        if bx < ox - 15 and b["xv"] < -140:
            bits &= ~GRAPPLE
        return bits
    # Avoid touching fast death ball directly; move beneath or jump over
    if dist < 55 and math.hypot(b["xv"], b["yv"]) > 130:
        bits |= (RIGHT if ox < WIDTH - 80 else LEFT)
        if o.get("onFloor", True) and by > oy - 40:
            bits |= UP | GRAPPLE
        return bits
    if bx > HALF_W - 20:
        bits |= LEFT if bx < ox - 12 else (RIGHT if bx > ox + 12 else 0)
        if 34 < dist < 95:
            bits |= GRAPPLE
    else:
        target_x = 550.0
        bits |= LEFT if target_x < ox - 15 else (RIGHT if target_x > ox + 15 else 0)
    return bits


class WdbGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage: int = 1, max_steps: int = 300, opponent_policy=None):
        super().__init__()
        self.stage = stage
        self.max_steps = max_steps
        self.opponent_policy = opponent_policy
        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_SIZE,), dtype=np.float32)

        bridge_path = Path(__file__).resolve().parent / "bridge.mjs"
        self.proc = subprocess.Popen(
            ["node", str(bridge_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.state = None
        self.steps = 0
        self.self_team = TEAM_RED
        self.opp_team = TEAM_BLUE
        self.mirror = False
        self.prev_action = 0

    def _rpc(self, payload: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("Node physics bridge closed unexpectedly")
        res = json.loads(line)
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "bridge error"))
        return res

    def _sample_overrides(self) -> dict | None:
        if self.stage == 1:
            # Stage 1 curriculum: ball starts loose on our side so bot learns grab & throw
            return {
                "ball": {
                    "x": random.uniform(140, 320),
                    "y": random.uniform(300, 420),
                    "xv": random.uniform(-40, 40),
                    "yv": 0,
                },
                "ftu": 0,
            }
        if self.stage == 2 and random.random() < 0.5:
            return {
                "ball": {
                    "x": random.uniform(220, 460),
                    "y": random.uniform(160, 340),
                    "xv": random.uniform(-120, 120),
                    "yv": random.uniform(-50, 50),
                },
                "ftu": 0,
            }
        return {"ball": {"x": HALF_W, "y": 230, "xv": random.uniform(-15, 15), "yv": 0}, "ftu": 15}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.prev_action = 0
        # Train both sides randomly so mirroring is fully symmetric
        self.mirror = bool(random.getrandbits(1))
        self.self_team = TEAM_BLUE if self.mirror else TEAM_RED
        self.opp_team = TEAM_RED if self.mirror else TEAM_BLUE

        players = [
            None,
            {"id": 1, "team": self.self_team},
            {"id": 2, "team": self.opp_team},
        ]
        res = self._rpc({
            "cmd": "reset",
            "players": players,
            "seed": seed or random.randint(1, 999999),
            "overrides": self._sample_overrides(),
        })
        self.state = res["state"]
        return build_obs(self.state, 1, 2, self.mirror), {}

    def step(self, action: int):
        self.steps += 1
        prev_state = self.state

        raw_bits = ACTION_TABLE[int(action)]
        bot_bits = mirror_bits(raw_bits) if self.mirror else raw_bits

        if self.opponent_policy is not None and self.stage >= 3:
            opp_obs = build_obs(self.state, 2, 1, not self.mirror)
            opp_act, _ = self.opponent_policy.predict(opp_obs, deterministic=False)
            opp_raw = ACTION_TABLE[int(opp_act)]
            opp_bits = mirror_bits(opp_raw) if (not self.mirror) else opp_raw
        elif self.stage == 1:
            opp_bits = 0  # Idle target in Stage 1
        else:
            opp_bits = scripted_opponent(self.state, 2, not self.mirror)

        res = self._rpc({"cmd": "step", "actions": {"1": bot_bits, "2": opp_bits}, "frames": 3})
        self.state = res["state"]
        winner = res["winner"]

        reward = self._compute_reward(prev_state, self.state, int(action), winner)
        terminated = winner is not None
        truncated = self.steps >= self.max_steps
        return build_obs(self.state, 1, 2, self.mirror), reward, terminated, truncated, {}

    def _compute_reward(self, prev: dict, curr: dict, action: int, winner: int | None) -> float:
        r = 0.0
        if winner == self.self_team:
            r += 1.0
        elif winner == self.opp_team:
            r -= 1.0

        me_prev = prev["discs"]["1"]
        me_curr = curr["discs"]["1"]
        b_curr = curr["ball"]

        # Reward grabbing/controlling the ball (only when alive)
        if me_curr["alive"] and me_curr["holdingBall"] and not me_prev["holdingBall"]:
            r += 0.08

        # Velocity toward enemy goal when ball is moving over (canonical: +x is toward enemy net)
        bxv = -b_curr["xv"] if self.mirror else b_curr["xv"]
        r += 0.012 * (bxv / BALL_MAX_V)

        # Tiny survival / anti-jitter penalty
        if (ACTION_TABLE[action] & (UP | DOWN)) != (ACTION_TABLE[self.prev_action] & (UP | DOWN)):
            r -= 0.002
        self.prev_action = action
        return float(r)

    def close(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            try:
                self._rpc({"cmd": "close"})
            except Exception:
                self.proc.kill()
        super().close()
