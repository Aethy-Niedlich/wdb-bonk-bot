import fs from 'node:fs';
import { box2d } from './bonk_init.mjs';
import { createBonkCore } from './bonk/bonk-core.mjs';
import { TEAM_RED, TEAM_BLUE } from './features.mjs';

export const PPM = 12;
export const bonk = createBonkCore(box2d, { thumbScale: 1 });
export const RAW_MAP = JSON.parse(fs.readFileSync(new URL('./bonk/WDB_live_map.json', import.meta.url), 'utf8'));

export const DEFAULT_SETTINGS = {
  map: RAW_MAP,
  gt: 2,
  wl: 9999,
  q: false,
  tl: false,
  tea: true,
  ga: 'b',
  mo: 'sp',
  bal: [],
};

export function maskToAct(m) {
  return {
    left: !!(m & 1),
    right: !!(m & 2),
    up: !!(m & 4),
    down: !!(m & 8),
    action: !!(m & 16),
    action2: !!(m & 32),
  };
}

export function findBallIdx(state) {
  const bodies = state?.physics?.bodies || [];
  const fx = state?.physics?.fixtures || [];
  let idx = bodies.findIndex((b) => b && b.s && b.s.n === 'DeathBall');
  if (idx < 0) {
    idx = bodies.findIndex((b) => b && b.s && b.s.type === 'd' && (b.fx || []).some((fi) => fx[fi]?.d === true));
  }
  return idx;
}

export class WdbEngine {
  constructor() {
    this.state = null;
    this.settings = DEFAULT_SETTINGS;
    this.ballIdx = -1;
    this.frame = 0;
    this.winner = null;
    this.lastBits = {};
  }

  reset(players = [], seed = 0, opts = null) {
    this.frame = 0;
    this.winner = null;
    this.lastBits = {};
    const gs = opts?.gs ? { ...DEFAULT_SETTINGS, ...opts.gs, map: RAW_MAP } : DEFAULT_SETTINGS;
    this.settings = gs;

    if (opts?.initialState) {
      this.state = opts.initialState;
    } else {
      const list = Array.isArray(players) ? players.filter(Boolean) : [];
      this.state = bonk.Physics.createNewState(list, RAW_MAP, seed | 0, false, null, gs, false);
      if (opts?.skipCountdown) {
        for (let k = 0; k < 95 && this.state && this.state.ftu > -1; k++) {
          this.state = bonk.Physics.prototype.step.call(bonk.Physics, this.state, [], null, 30, gs, 1);
        }
      }
    }
    this.ballIdx = findBallIdx(this.state);
    return this.getState();
  }

  step(actions = {}, frames = 1) {
    for (let f = 0; f < frames; f++) {
      this.frame++;
      if (!this.state || !this.state.discs) break;

      const prevFte = this.state.fte;
      const prevS2 = this.state.scores?.[TEAM_RED] ?? 0;
      const prevS3 = this.state.scores?.[TEAM_BLUE] ?? 0;

      const inp = new Array(this.state.discs.length);
      for (let slot = 0; slot < this.state.discs.length; slot++) {
        if (!this.state.discs[slot]) continue;
        const raw = actions[String(slot)] ?? actions[slot] ?? this.lastBits[slot] ?? 0;
        const bits = Number(raw) || 0;
        this.lastBits[slot] = bits;
        inp[slot] = maskToAct(bits);
      }

      this.state = bonk.Physics.prototype.step.call(bonk.Physics, this.state, inp, null, 30, this.settings, 1);
      if (!this.state || !this.state.discs) break;

      if (prevFte === -1 && this.state.fte >= 0 && this.winner === null) {
        const s2 = this.state.scores?.[TEAM_RED] ?? 0;
        const s3 = this.state.scores?.[TEAM_BLUE] ?? 0;
        if (s2 > prevS2) this.winner = TEAM_RED;
        else if (s3 > prevS3) this.winner = TEAM_BLUE;
        else {
          const redAlive = this.state.discs.some((d) => d && d.team === TEAM_RED && d.ds === 0);
          const blueAlive = this.state.discs.some((d) => d && d.team === TEAM_BLUE && d.ds === 0);
          if (redAlive && !blueAlive) this.winner = TEAM_RED;
          else if (blueAlive && !redAlive) this.winner = TEAM_BLUE;
          else {
            const bb = this.ballIdx >= 0 ? this.state.physics?.bodies?.[this.ballIdx] : null;
            if (bb) this.winner = bb.p[0] < 30.417 ? TEAM_RED : TEAM_BLUE;
          }
        }
      }
    }
    return { state: this.getState(), frame: this.frame, winner: this.winner };
  }

  getRawState() {
    return structuredClone(this.state);
  }

  setRawState(rawState) {
    this.state = structuredClone(rawState);
    this.ballIdx = findBallIdx(this.state);
  }

  getState() {
    const s = this.state;
    const bb = this.ballIdx >= 0 ? s?.physics?.bodies?.[this.ballIdx] : null;
    const discs = {};
    if (s?.discs) {
      for (let i = 0; i < s.discs.length; i++) {
        const d = s.discs[i];
        if (!d) continue;
        const py = d.y * PPM;
        discs[String(i)] = {
          x: d.x * PPM,
          y: py,
          xv: d.xv * PPM,
          yv: d.yv * PPM,
          team: d.team,
          alive: d.ds === 0,
          heavy: !!d.a1,
          stamina: d.a1a ?? 1000,
          grappleMs: ((d.a1a ?? 1000) / 1000) * 5000,
          onFloor: py >= 446 && Math.abs(d.yv * PPM) < 25,
          holdingBall: !!(d.swing && d.swing.b === this.ballIdx),
        };
      }
    }
    return {
      ball: bb
        ? { x: bb.p[0] * PPM, y: bb.p[1] * PPM, xv: bb.lv[0] * PPM, yv: bb.lv[1] * PPM }
        : { x: 365, y: 250, xv: 0, yv: 0 },
      discs,
      ftu: Math.max(0, s?.ftu ?? 0),
      fte: s?.fte ?? -1,
      scores: Array.isArray(s?.scores) ? s.scores.slice() : [],
    };
  }
}
