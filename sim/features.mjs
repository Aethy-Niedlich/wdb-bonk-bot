export const WIDTH = 730;
export const HEIGHT = 500;
export const HALF_W = 365;
export const HALF_H = 250;
export const LEFT_WALL_X = 35;
export const RIGHT_WALL_X = 695;
export const FLOOR_Y = 462.5;
export const GOAL_Y = 408;
export const PLAYER_MAX_V = 400;
export const BALL_MAX_V = 600;

export const TEAM_RED = 2;
export const TEAM_BLUE = 3;

export const LEFT = 1;
export const RIGHT = 2;
export const UP = 4;
export const DOWN = 8;
export const HEAVY = 16;
export const GRAPPLE = 32;

// 27 clean actions: [0, L, R] x [0, U, D] x [0, Heavy, Grapple]
export const ACTION_TABLE = (() => {
  const out = [];
  for (const h of [0, LEFT, RIGHT]) {
    for (const v of [0, UP, DOWN]) {
      for (const m of [0, HEAVY, GRAPPLE]) {
        out.push(h | v | m);
      }
    }
  }
  return out;
})();

const BIT_TO_IDX = new Map(ACTION_TABLE.map((b, i) => [b, i]));

export function bitsToActionIndex(bits) {
  let b = bits & 63;
  // Clean conflicting L+R or U+D or simultaneous Heavy+Grapple from human replays
  if ((b & (LEFT | RIGHT)) === (LEFT | RIGHT)) b &= ~(LEFT | RIGHT);
  if ((b & (UP | DOWN)) === (UP | DOWN)) b &= ~(UP | DOWN);
  if ((b & (HEAVY | GRAPPLE)) === (HEAVY | GRAPPLE)) b &= ~HEAVY;
  return BIT_TO_IDX.get(b) ?? 0;
}

export function mirrorBits(bits) {
  let rest = bits & ~(LEFT | RIGHT);
  if (bits & LEFT) rest |= RIGHT;
  if (bits & RIGHT) rest |= LEFT;
  return rest;
}

export const OBS_SIZE = 30;

export function buildObs(state, selfId, oppId, mirror) {
  const b = state.ball;
  const d = state.discs[String(selfId)];
  const o = oppId != null ? state.discs[String(oppId)] : null;

  const cx = (x) => (mirror ? WIDTH - x : x);
  const cvx = (xv) => (mirror ? -xv : xv);

  const sx = cx(d?.x ?? 265), sy = d?.y ?? FLOOR_Y - 12;
  const sxv = cvx(d?.xv ?? 0), syv = d?.yv ?? 0;
  const bx = cx(b.x), by = b.y;
  const bxv = cvx(b.xv), byv = b.yv;
  const ox = o?.alive ? cx(o.x) : HALF_W, oy = o?.alive ? o.y : HALF_H;
  const oxv = cvx(o?.xv ?? 0), oyv = o?.yv ?? 0;

  const vals = [
    (sx - HALF_W) / HALF_W,
    (sy - HALF_H) / HALF_H,
    sxv / PLAYER_MAX_V,
    syv / PLAYER_MAX_V,
    (d?.grappleMs ?? 5000) / 5000,
    (d?.stamina ?? 1000) / 1000,
    d?.onFloor ?? true ? 1 : 0,
    d?.heavy ? 1 : 0,
    d?.holdingBall ? 1 : 0,
    (bx - HALF_W) / HALF_W,
    (by - HALF_H) / HALF_H,
    bxv / BALL_MAX_V,
    byv / BALL_MAX_V,
    (bx - sx) / HALF_W,
    (by - sy) / HALF_H,
    (bxv - sxv) / BALL_MAX_V,
    (byv - syv) / BALL_MAX_V,
    Math.hypot(bx - sx, by - sy) / 400,
    (ox - HALF_W) / HALF_W,
    (oy - HALF_H) / HALF_H,
    oxv / PLAYER_MAX_V,
    oyv / PLAYER_MAX_V,
    o?.holdingBall ? 1 : 0,
    o?.alive ?? false ? 1 : 0,
    (RIGHT_WALL_X - bx) / WIDTH,
    (GOAL_Y - by) / HEIGHT,
    (bx - LEFT_WALL_X) / WIDTH,
    (GOAL_Y - by) / HEIGHT,
    bx < HALF_W ? 1 : 0,
    Math.min(Math.max(state.ftu ?? 0, 0), 90) / 90,
  ];

  const out = new Float32Array(OBS_SIZE);
  for (let i = 0; i < OBS_SIZE; i++) {
    out[i] = Math.max(-5, Math.min(5, Math.fround(vals[i])));
  }
  return out;
}
