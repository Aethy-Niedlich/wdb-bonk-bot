// sim/wdb_engine.mjs
import { World, Vec2, Circle, Edge, DistanceJoint } from 'planck';

export const WIDTH = 730;
export const HEIGHT = 500;
export const PPM = 12; // bonk.io map pixels-per-meter
export const FPS = 30;
export const DT = 1 / FPS;

export const MID_X = 365;
export const LEFT_WALL_X = 14;
export const RIGHT_WALL_X = 716;
export const CEIL_Y = 14;
export const FLOOR_Y = 468;
export const GOAL_TOP_Y = 365;

export const PLAYER_RADIUS = 12;
export const BALL_RADIUS = 18;
export const GRAPPLE_RANGE = 110;
export const GRAPPLE_MAX_MS = 5000;
export const STAMINA_MAX = 1000;

export const TEAM_RED = 2;
export const TEAM_BLUE = 3;

export const LEFT = 1;
export const RIGHT = 2;
export const UP = 4;
export const DOWN = 8;
export const HEAVY = 16;
export const GRAPPLE = 32;

const CAT_PLAYER = 0x0001;
const CAT_BALL = 0x0002;
const CAT_WALL = 0x0004;
const CAT_BARRIER = 0x0008;

const m = (px) => px / PPM;
const px = (meters) => meters * PPM;

// Static surfaces in WDB pixels (used both for Box2D fixtures and grapple targeting)
const WALL_SEGMENTS = [
  { x1: LEFT_WALL_X, y1: CEIL_Y, x2: RIGHT_WALL_X, y2: CEIL_Y, isFloor: false, restitution: 0.35 },
  { x1: LEFT_WALL_X, y1: FLOOR_Y, x2: RIGHT_WALL_X, y2: FLOOR_Y, isFloor: true, restitution: 0.35 },
  { x1: LEFT_WALL_X, y1: CEIL_Y, x2: LEFT_WALL_X, y2: GOAL_TOP_Y, isFloor: false, restitution: 0.4 },
  { x1: RIGHT_WALL_X, y1: CEIL_Y, x2: RIGHT_WALL_X, y2: GOAL_TOP_Y, isFloor: false, restitution: 0.4 },
  // Center kickoff mound
  { x1: MID_X - 22, y1: FLOOR_Y, x2: MID_X, y2: FLOOR_Y - 22, isFloor: true, restitution: 0.45 },
  { x1: MID_X, y1: FLOOR_Y - 22, x2: MID_X + 22, y2: FLOOR_Y, isFloor: true, restitution: 0.45 },
  // Angled upper bouncers (characteristic of WDB)
  { x1: LEFT_WALL_X, y1: 185, x2: 52, y2: 135, isFloor: false, restitution: 1.3 },
  { x1: RIGHT_WALL_X, y1: 185, x2: 678, y2: 135, isFloor: false, restitution: 1.3 },
];

function closestPointOnSegment(qx, qy, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-8) return { x: x1, y: y1, dist: Math.hypot(qx - x1, qy - y1) };
  const t = Math.max(0, Math.min(1, ((qx - x1) * dx + (qy - y1) * dy) / lenSq));
  const cx = x1 + t * dx;
  const cy = y1 + t * dy;
  return { x: cx, y: cy, dist: Math.hypot(qx - cx, qy - cy) };
}

export class WdbEngine {
  constructor() {
    this.world = null;
    this.wallBody = null;
    this.ballBody = null;
    this.players = new Map();
    this.pendingDeaths = new Set();
    this.frame = 0;
    this.ftu = 0;
    this.winner = null;
  }

  reset(players = [], seed = 0, overrides = null) {
    this.world = new World({ gravity: Vec2(0, 25) });
    this.players.clear();
    this.pendingDeaths.clear();
    this.frame = 0;
    this.ftu = overrides?.ftu ?? 0;
    this.winner = null;

    // Static map body (walls, floor, ceiling, bouncers)
    this.wallBody = this.world.createBody();
    for (const s of WALL_SEGMENTS) {
      this.wallBody.createFixture(Edge(Vec2(m(s.x1), m(s.y1)), Vec2(m(s.x2), m(s.y2))), {
        friction: 0.35,
        restitution: s.restitution,
        filterCategoryBits: CAT_WALL,
        filterMaskBits: CAT_PLAYER | CAT_BALL,
      });
    }

    // Middle barrier: blocks players, allows the death ball to pass through
    this.wallBody.createFixture(Edge(Vec2(m(MID_X), m(CEIL_Y)), Vec2(m(MID_X), m(FLOOR_Y))), {
      friction: 0.1,
      restitution: 0.3,
      filterCategoryBits: CAT_BARRIER,
      filterMaskBits: CAT_PLAYER,
    });

    // Death ball
    const bInit = overrides?.ball ?? { x: MID_X, y: 240, xv: 0, yv: 0 };
    this.ballBody = this.world.createDynamicBody({
      position: Vec2(m(bInit.x), m(bInit.y)),
      linearVelocity: Vec2(m(bInit.xv ?? 0), m(bInit.yv ?? 0)),
      linearDamping: 0.03,
      fixedRotation: true,
      bullet: true,
    });
    this.ballBody.createFixture(Circle(m(BALL_RADIUS)), {
      density: 0.8,
      friction: 0.2,
      restitution: 0.88,
      filterCategoryBits: CAT_BALL,
      filterMaskBits: CAT_WALL | CAT_PLAYER,
    });
    this.ballBody.setUserData({ kind: 'ball' });

    // Players (sparse array or list of { id, team })
    const list = Array.isArray(players) ? players.filter(Boolean) : [];
    for (const p of list) {
      if (p.team !== TEAM_RED && p.team !== TEAM_BLUE) continue;
      const ov = overrides?.discs?.[String(p.id)];
      const spawnX = ov?.x ?? (p.team === TEAM_RED ? 165 : WIDTH - 165);
      const spawnY = ov?.y ?? FLOOR_Y - PLAYER_RADIUS - 2;

      const body = this.world.createDynamicBody({
        position: Vec2(m(spawnX), m(spawnY)),
        linearVelocity: Vec2(m(ov?.xv ?? 0), m(ov?.yv ?? 0)),
        linearDamping: 0.45,
        fixedRotation: true,
        bullet: true,
      });
      const fixture = body.createFixture(Circle(m(PLAYER_RADIUS)), {
        density: 1.0,
        friction: 0.35,
        restitution: 0.25,
        filterCategoryBits: CAT_PLAYER,
        filterMaskBits: CAT_WALL | CAT_BARRIER | CAT_PLAYER | CAT_BALL,
      });
      body.setUserData({ kind: 'player', id: p.id });

      this.players.set(p.id, {
        id: p.id,
        team: p.team,
        body,
        fixture,
        alive: true,
        heavy: false,
        stamina: STAMINA_MAX,
        grappleMs: GRAPPLE_MAX_MS,
        grappleLocked: false,
        joint: null,
        grappleTarget: null, // 'ball' | 'wall' | null
        jumpCooldown: 0,
        lastBits: 0,
      });
    }

    // Death ball kills players on contact (processed safely after world.step)
    this.world.on('begin-contact', (contact) => {
      const a = contact.getFixtureA().getBody().getUserData();
      const b = contact.getFixtureB().getBody().getUserData();
      if (!a || !b) return;
      if (a.kind === 'ball' && b.kind === 'player') this.pendingDeaths.add(b.id);
      if (b.kind === 'ball' && a.kind === 'player') this.pendingDeaths.add(a.id);
    });

    this.world.on('remove-joint', (joint) => {
      for (const p of this.players.values()) {
        if (p.joint === joint) {
          p.joint = null;
          p.grappleTarget = null;
        }
      }
    });

    return this.getState();
  }

  _releaseJoint(p) {
    if (p.joint) {
      this.world.destroyJoint(p.joint);
      p.joint = null;
    }
    p.grappleTarget = null;
  }

  _tryGrapple(p, bits) {
    const pos = p.body.getPosition();
    const pxX = px(pos.x);
    const pxY = px(pos.y);

    // Distance to death ball surface
    const bPos = this.ballBody.getPosition();
    const bPxX = px(bPos.x);
    const bPxY = px(bPos.y);
    const ballSurfaceDist = Math.hypot(bPxX - pxX, bPxY - pxY) - BALL_RADIUS;

    // Nearest static surface
    let bestWall = null;
    let bestWallDist = Infinity;
    for (const s of WALL_SEGMENTS) {
      const hit = closestPointOnSegment(pxX, pxY, s.x1, s.y1, s.x2, s.y2);
      if (hit.dist < bestWallDist) {
        bestWallDist = hit.dist;
        bestWall = { ...hit, isFloor: s.isFloor };
      }
    }
    // Also check barrier as grappleable surface
    const barrierHit = closestPointOnSegment(pxX, pxY, MID_X, CEIL_Y, MID_X, FLOOR_Y);
    if (barrierHit.dist < bestWallDist) {
      bestWallDist = barrierHit.dist;
      bestWall = { ...barrierHit, isFloor: false };
    }

    // Grab the nearest object in grappleRange (with slight preference when ball is clear)
    if (ballSurfaceDist <= GRAPPLE_RANGE && ballSurfaceDist <= bestWallDist - 2) {
      const distMeters = Math.max(pos.clone().sub(bPos).length(), m(PLAYER_RADIUS + BALL_RADIUS + 6));
      p.joint = this.world.createJoint(
        new DistanceJoint(
          {
            length: distMeters,
            frequencyHz: 4.5,
            dampingRatio: 0.35,
            collideConnected: true,
          },
          p.body,
          this.ballBody,
          pos,
          bPos,
        ),
      );
      p.grappleTarget = 'ball';
      p.grappleMs = Math.max(0, p.grappleMs - 1000 / FPS);
      return;
    }

    if (bestWall && bestWallDist <= GRAPPLE_RANGE) {
      if (bestWall.isFloor && p.jumpCooldown === 0 && pxY >= FLOOR_Y - 55) {
        // WDB moving-floor bounce: stronger near side edges and when holding Heavy (X)
        const edgeFactor = Math.min(1, Math.abs(pxX - MID_X) / (MID_X - LEFT_WALL_X));
        const heavyBoost = p.heavy ? 1.32 : 1.0;
        const upSpeedPx = (310 + 195 * edgeFactor) * heavyBoost;
        const vel = p.body.getLinearVelocity();
        p.body.setLinearVelocity(Vec2(vel.x, Math.min(vel.y, -m(upSpeedPx))));
        p.jumpCooldown = 8;
        p.grappleMs = Math.max(0, p.grappleMs - 140);
      } else if (!bestWall.isFloor) {
        const anchor = Vec2(m(bestWall.x), m(bestWall.y));
        const distMeters = Math.max(pos.clone().sub(anchor).length(), m(16));
        p.joint = this.world.createJoint(
          new DistanceJoint(
            {
              length: distMeters,
              frequencyHz: 5.0,
              dampingRatio: 0.4,
              collideConnected: true,
            },
            p.body,
            this.wallBody,
            pos,
            anchor,
          ),
        );
        p.grappleTarget = 'wall';
        p.grappleMs = Math.max(0, p.grappleMs - 1000 / FPS);
      }
    }
  }

  step(actions = {}, frames = 1) {
    for (let f = 0; f < frames; f++) {
      this.frame++;
      if (this.winner !== null) break;

      if (this.ftu > 0) {
        this.ftu--;
        continue;
      }

      for (const p of this.players.values()) {
        if (!p.alive) continue;
        const bits = Number(actions[String(p.id)] ?? actions[p.id] ?? p.lastBits) || 0;
        p.lastBits = bits;
        if (p.jumpCooldown > 0) p.jumpCooldown--;

        // Heavy (X) stamina: -10/frame held, +5/frame released
        const wantsHeavy = !!(bits & HEAVY);
        if (wantsHeavy && p.stamina > 0) {
          p.stamina = Math.max(0, p.stamina - 10);
          p.heavy = true;
        } else {
          p.stamina = Math.min(STAMINA_MAX, p.stamina + 5);
          p.heavy = false;
        }

        const targetDensity = p.heavy ? 3.2 : 1.0;
        if (p.fixture.getDensity() !== targetDensity) {
          p.fixture.setDensity(targetDensity);
          p.body.resetMassData();
        }

        // Movement forces
        const pos = p.body.getPosition();
        const onFloor = px(pos.y) >= FLOOR_Y - PLAYER_RADIUS - 3;
        const moveAcc = onFloor ? 34 : 24;
        let fx = 0;
        let fy = 0;
        if (bits & LEFT) fx -= moveAcc;
        if (bits & RIGHT) fx += moveAcc;
        if (bits & UP) fy -= onFloor ? 16 : 18;
        if (bits & DOWN) fy += 24;
        if (p.heavy) fy += 18;

        const mass = p.body.getMass();
        p.body.applyForceToCenter(Vec2(fx * mass, fy * mass), true);

        // Soft speed clamp (px/s -> m/s)
        const v = p.body.getLinearVelocity();
        const maxVx = m(330);
        if (Math.abs(v.x) > maxVx) {
          p.body.setLinearVelocity(Vec2(Math.sign(v.x) * maxVx, v.y));
        }

        // Grapple (Z / SPECIAL)
        const wantsGrapple = !!(bits & GRAPPLE);
        if (wantsGrapple && !p.grappleLocked && p.grappleMs > 0) {
          if (p.joint) {
            p.grappleMs = Math.max(0, p.grappleMs - 1000 / FPS);
            if (p.grappleMs === 0) {
              p.grappleLocked = true;
              this._releaseJoint(p);
            }
          } else {
            this._tryGrapple(p, bits);
            if (p.grappleMs === 0) p.grappleLocked = true;
          }
        } else {
          this._releaseJoint(p);
          p.grappleMs = Math.min(GRAPPLE_MAX_MS, p.grappleMs + 1000 / FPS);
          if (p.grappleLocked && p.grappleMs >= GRAPPLE_MAX_MS) {
            p.grappleLocked = false;
          }
        }
      }

      this.world.step(DT, 8, 3);

      // Process deaths outside world.step
      if (this.pendingDeaths.size > 0) {
        for (const id of this.pendingDeaths) {
          const p = this.players.get(id);
          if (!p || !p.alive) continue;
          p.alive = false;
          this._releaseJoint(p);
          p.body.setActive(false);
        }
        this.pendingDeaths.clear();
      }

      // Check round win conditions (goal or death)
      const bPos = this.ballBody.getPosition();
      const bx = px(bPos.x);
      const by = px(bPos.y);
      if (bx < LEFT_WALL_X && by > GOAL_TOP_Y) {
        this.winner = TEAM_BLUE;
      } else if (bx > RIGHT_WALL_X && by > GOAL_TOP_Y) {
        this.winner = TEAM_RED;
      } else {
        let redAlive = 0;
        let blueAlive = 0;
        for (const p of this.players.values()) {
          if (!p.alive) continue;
          if (p.team === TEAM_RED) redAlive++;
          if (p.team === TEAM_BLUE) blueAlive++;
        }
        if (redAlive === 0 && blueAlive > 0) this.winner = TEAM_BLUE;
        else if (blueAlive === 0 && redAlive > 0) this.winner = TEAM_RED;
      }
    }

    return { state: this.getState(), frame: this.frame, winner: this.winner };
  }

  getState() {
    const bp = this.ballBody.getPosition();
    const bv = this.ballBody.getLinearVelocity();
    const discs = {};
    for (const p of this.players.values()) {
      const pos = p.body.getPosition();
      const vel = p.body.getLinearVelocity();
      const py = px(pos.y);
      discs[String(p.id)] = {
        x: px(pos.x),
        y: py,
        xv: px(vel.x),
        yv: px(vel.y),
        team: p.team,
        alive: p.alive,
        heavy: p.heavy,
        stamina: p.stamina,
        grappleMs: p.grappleMs,
        onFloor: py >= FLOOR_Y - PLAYER_RADIUS - 3,
        holdingBall: p.grappleTarget === 'ball',
      };
    }
    return {
      ball: { x: px(bp.x), y: px(bp.y), xv: px(bv.x), yv: px(bv.y) },
      discs,
      ftu: this.ftu,
    };
  }
}
