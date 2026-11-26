// bot/features.ts
export const WIDTH = 730;
export const HEIGHT = 500;
export const HALF_W = 365;
export const HALF_H = 250;
export const LEFT_WALL_X = 14;
export const RIGHT_WALL_X = 716;
export const GOAL_Y = 420;
export const FLOOR_Y = 468;
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

export const ACTION_TABLE: readonly number[] = (() => {
  const out: number[] = [];
  for (const h of [0, LEFT, RIGHT]) {
    for (const v of [0, UP, DOWN]) {
      for (const m of [0, HEAVY, GRAPPLE, HEAVY | GRAPPLE]) {
        out.push(h | v | m);
      }
    }
  }
  return out;
})();

export const OBS_SIZE = 30;

export function mirrorBits(bits: number): number {
  let rest = bits & ~(LEFT | RIGHT);
  if (bits & LEFT) rest |= RIGHT;
  if (bits & RIGHT) rest |= LEFT;
  return rest;
}

export interface DiscState {
  x: number;
  y: number;
  xv: number;
  yv: number;
  team?: number;
  alive?: boolean;
  heavy?: boolean;
  stamina?: number;
  grappleMs?: number;
  onFloor?: boolean;
  holdingBall?: boolean;
}

export interface WdbState {
  ball: { x: number; y: number; xv: number; yv: number };
  discs: Record<string, DiscState>;
  ftu: number;
}

export function buildObs(state: WdbState, selfId: number, oppId: number, mirror: boolean): Float32Array {
  const b = state.ball;
  const d = state.discs[String(selfId)];
  const o = state.discs[String(oppId)];

  const cx = (x: number) => (mirror ? WIDTH - x : x);
  const cvx = (xv: number) => (mirror ? -xv : xv);

  const sx = cx(d?.x ?? 165), sy = d?.y ?? FLOOR_Y - 14;
  const sxv = cvx(d?.xv ?? 0), syv = d?.yv ?? 0;
  const bx = cx(b.x), by = b.y;
  const bxv = cvx(b.xv), byv = b.yv;
  const ox = cx(o?.x ?? WIDTH - 165), oy = o?.y ?? FLOOR_Y - 14;
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
    o?.alive ?? true ? 1 : 0,
    (RIGHT_WALL_X - bx) / WIDTH,
    (GOAL_Y - by) / HEIGHT,
    (bx - LEFT_WALL_X) / WIDTH,
    (GOAL_Y - by) / HEIGHT,
    bx < HALF_W ? 1 : 0,
    Math.min(Math.max(state.ftu ?? 0, 0), 90) / 90,
  ];

  const out = new Float32Array(OBS_SIZE);
  for (let i = 0; i < OBS_SIZE; i++) {
    out[i] = Math.max(-5, Math.min(5, Math.fround(vals[i]!)));
  }
  return out;
}
