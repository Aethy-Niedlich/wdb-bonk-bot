import fs from 'node:fs';
import path from 'node:path';
import { bonk, findBallIdx, PPM } from '../sim/wdb_engine.mjs';
import {
  buildObs, bitsToActionIndex, mirrorBits, ACTION_TABLE,
  TEAM_RED, TEAM_BLUE, LEFT, RIGHT, UP, DOWN, HEAVY, GRAPPLE,
} from '../sim/features.mjs';

const TOOL_DIR = process.argv[2] || 'data/tool_logs';
const REPLAY_DIR = process.argv[3] || '/content/wdb-bot/reference/replays';

const CODE_TO_BIT = {
  KeyA: LEFT, ArrowLeft: LEFT,
  KeyD: RIGHT, ArrowRight: RIGHT,
  KeyW: UP, ArrowUp: UP,
  KeyS: DOWN, ArrowDown: DOWN,
  Space: HEAVY, KeyX: HEAVY, ShiftLeft: HEAVY, ShiftRight: HEAVY,
  KeyT: GRAPPLE, KeyZ: GRAPPLE, KeyY: GRAPPLE,
};

const obsList = [];
const actList = [];

function pushSample(ws, slot, oppId, mirror, bits) {
  obsList.push(buildObs(ws, slot, oppId, mirror));
  const canonBits = mirror ? mirrorBits(bits) : bits;
  actList.push(bitsToActionIndex(canonBits));
}

function isPlayerRec(b) {
  if (b[6]) return b[6] === 'disc' && b[5] !== null && Math.abs(b[5] - 1) < 0.2;
  return b[0] !== 4 && b[0] !== 6 && b[5] !== null && Math.abs(b[5] - 1) < 0.2;
}

function findBallRec(bodies) {
  return bodies.find((x) => x[9] === 1) || bodies.find((x) => x[0] === 4 && (!x[6] || x[6] !== 'disc')) || null;
}

function parseToolLog(filePath) {
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  let meta = null;
  const ticks = [];
  const keyEvents = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    let obj;
    try { obj = JSON.parse(line); } catch { continue; }
    if (obj.type === 'meta') { meta = obj; continue; }
    if (obj.b) ticks.push(obj);
    else if (obj.k && !obj.bot) keyEvents.push(obj);
  }
  if (ticks.length < 30 || keyEvents.length === 0) return 0;

  let meId = meta?.defence?.me ?? null;
  if (meId == null) {
    const byN = new Map(ticks.map((t) => [t.n, t]));
    const votes = {};
    for (const ke of keyEvents) {
      if (ke.k !== 'down') continue;
      const dir = (ke.code === 'KeyA' || ke.code === 'ArrowLeft') ? -1
        : (ke.code === 'KeyD' || ke.code === 'ArrowRight') ? 1 : 0;
      if (!dir) continue;
      const t0 = byN.get(ke.n);
      const t1 = byN.get(ke.n + 4);
      if (!t0 || !t1) continue;
      const p0 = new Map(t0.b.filter(isPlayerRec).map((x) => [x[0], x]));
      let best = null, second = -1e9;
      for (const b1 of t1.b) {
        if (!isPlayerRec(b1)) continue;
        const b0 = p0.get(b1[0]);
        if (!b0) continue;
        const gain = (b1[3] - b0[3]) * dir;
        if (!best || gain > best.gain) { if (best) second = Math.max(second, best.gain); best = { id: b1[0], gain }; }
        else second = Math.max(second, gain);
      }
      if (best && best.gain > 0.8 && best.gain > second + 0.5) {
        votes[best.id] = (votes[best.id] || 0) + 1;
      }
    }
    const ranked = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    if (ranked.length) meId = Number(ranked[0][0]);
  }
  if (meId == null) {
    console.warn(`Skipping ${path.basename(filePath)}: could not identify your body ID.`);
    return 0;
  }

  const heldCodes = new Set();
  let ki = 0;
  let a1a = 1000;
  let wasMissing = true;
  let myTeam = TEAM_BLUE;
  let added = 0;

  for (const tk of ticks) {
    while (ki < keyEvents.length && keyEvents[ki].n < tk.n) {
      const ke = keyEvents[ki++];
      if (ke.k === 'sync' && Array.isArray(ke.held)) {
        heldCodes.clear();
        for (const c of ke.held) if (CODE_TO_BIT[c]) heldCodes.add(c);
      } else if (CODE_TO_BIT[ke.code]) {
        if (ke.k === 'down') heldCodes.add(ke.code);
        else heldCodes.delete(ke.code);
      }
    }

    const ball = findBallRec(tk.b);
    const me = tk.b.find((x) => x[0] === meId && (!x[6] || x[6] === 'disc'));
    if (!me || !ball) { wasMissing = true; continue; }

    if (wasMissing) {
      a1a = 1000;
      myTeam = (me[7] === TEAM_RED || me[7] === TEAM_BLUE) ? me[7] : (me[1] < 30.417 ? TEAM_BLUE : TEAM_RED);
      wasMissing = false;
    }

    const joints = tk.j || [];
    const swinging = joints.some((j) => j[0] === 3 && ((j[1] === meId && j[4] !== 'phys') || (j[2] === meId && j[5] !== 'phys')));
    if (swinging) {
      a1a = Math.max(0, a1a - 4);
      if (a1a < 500) a1a = 0;
    } else {
      a1a = Math.min(1000, a1a + 3);
    }

    const discs = {};
    let oppId = null, bestDist = Infinity;
    for (const b of tk.b) {
      if (!isPlayerRec(b)) continue;
      const py = b[2] * PPM;
      const team = (b[7] === TEAM_RED || b[7] === TEAM_BLUE) ? b[7] : (b[1] < 30.417 ? TEAM_BLUE : TEAM_RED);
      const hb = joints.some((j) => j[0] === 3 && (
        (j[1] === b[0] && j[4] !== 'phys' && j[2] === ball[0] && j[5] !== 'disc') ||
        (j[2] === b[0] && j[5] !== 'phys' && j[1] === ball[0] && j[4] !== 'disc')
      ));
      const isHeavy = b[8] ? b[8] > 1.05 : false;
      const estA1a = (b[0] === meId) ? a1a : (isHeavy ? Math.min(1000, Math.max(0, ((b[8] - 1) / 3.7) * 1000)) : 1000);
      discs[String(b[0])] = {
        x: b[1] * PPM, y: py, xv: b[3] * PPM, yv: b[4] * PPM,
        team, alive: true, heavy: isHeavy,
        stamina: estA1a, grappleMs: (estA1a / 1000) * 5000,
        onFloor: py >= 446 && Math.abs(b[4] * PPM) < 25,
        holdingBall: hb,
      };
      if (b[0] !== meId && team !== myTeam) {
        const dist = Math.hypot(b[1] - ball[1], b[2] - ball[2]);
        if (dist < bestDist) { bestDist = dist; oppId = b[0]; }
      }
    }

    let bits = 0;
    for (const code of heldCodes) bits |= (CODE_TO_BIT[code] || 0);
    if (bits === 0 && Math.hypot(me[3], me[4]) < 0.05 && Math.hypot(ball[3], ball[4]) < 0.05) continue;

    pushSample({
      ball: { x: ball[1] * PPM, y: ball[2] * PPM, xv: ball[3] * PPM, yv: ball[4] * PPM },
      discs, ftu: 0,
    }, meId, oppId, myTeam === TEAM_RED, bits);
    added++;
  }
  console.log(`Tool log ${path.basename(filePath)}: me=id${meId}, added ${added} frames`);
  return added;
}

if (fs.existsSync(TOOL_DIR)) {
  for (const f of fs.readdirSync(TOOL_DIR).filter((x) => x.endsWith('.jsonl')).sort()) {
    parseToolLog(path.join(TOOL_DIR, f));
  }
}

function decodeInputs(b64) {
  const buf = Buffer.from(b64, 'base64');
  let off = 0;
  const count = buf.readUInt16BE(off); off += 2;
  const byTick = [];
  for (let i = 0; i < count; i++) {
    const slot = buf.readUInt16BE(off); off += 2;
    const tick = buf.readUInt32BE(off); off += 4;
    const f = buf.readUInt8(off); off += 1;
    (byTick[tick] ||= [])[slot] = {
      left: !!(f & 1), right: !!(f & 2), up: !!(f & 4),
      down: !!(f & 8), action: !!(f & 16), action2: !!(f & 32), bits: f & 63,
    };
  }
  return byTick;
}

if (fs.existsSync(REPLAY_DIR)) {
  const files = fs.readdirSync(REPLAY_DIR).filter((f) => f.endsWith('.json')).sort();
  let replayFrames = 0;
  for (const file of files) {
    const replay = JSON.parse(fs.readFileSync(path.join(REPLAY_DIR, file), 'utf8'));
    const byTick = decodeInputs(replay.inputs);
    let s = replay.startingState;
    const ballIdx = findBallIdx(s);
    const held = [];
    for (let t = 0; t < byTick.length; t++) {
      if (!s || !s.discs || s.fte === 0) break;
      if (byTick[t]) for (const k in byTick[t]) held[k] = byTick[t][k];
      const prevFte = s.fte;
      if (s.ftu === -1 && prevFte === -1) {
        const bb = ballIdx >= 0 ? s.physics?.bodies?.[ballIdx] : null;
        const discs = {};
        for (let i = 0; i < s.discs.length; i++) {
          const d = s.discs[i];
          if (!d) continue;
          const py = d.y * PPM;
          discs[String(i)] = {
            x: d.x * PPM, y: py, xv: d.xv * PPM, yv: d.yv * PPM,
            team: d.team, alive: d.ds === 0, heavy: !!d.a1,
            stamina: d.a1a ?? 1000, grappleMs: ((d.a1a ?? 1000) / 1000) * 5000,
            onFloor: py >= 446 && Math.abs(d.yv * PPM) < 25,
            holdingBall: !!(d.swing && d.swing.b === ballIdx),
          };
        }
        const ws = {
          ball: bb ? { x: bb.p[0] * PPM, y: bb.p[1] * PPM, xv: bb.lv[0] * PPM, yv: bb.lv[1] * PPM } : { x: 365, y: 250, xv: 0, yv: 0 },
          discs, ftu: 0,
        };
        for (let slot = 0; slot < s.discs.length; slot++) {
          const d = s.discs[slot], act = held[slot];
          if (!d || d.ds !== 0 || !act || (d.team !== TEAM_RED && d.team !== TEAM_BLUE)) continue;
          let oppId = null, bestDist = Infinity;
          for (let j = 0; j < s.discs.length; j++) {
            const o = s.discs[j];
            if (!o || o.ds !== 0 || o.team === d.team || (o.team !== TEAM_RED && o.team !== TEAM_BLUE)) continue;
            const dist = Math.hypot(o.x * PPM - ws.ball.x, o.y * PPM - ws.ball.y);
            if (dist < bestDist) { bestDist = dist; oppId = j; }
          }
          pushSample(ws, slot, oppId, d.team === TEAM_RED, act.bits);
          replayFrames++;
        }
      }
      s = bonk.Physics.prototype.step.call(bonk.Physics, s, held, replay.adminInputs?.[t] || null, 30, replay.gameSettings, 1);
      if (prevFte === -1 && s?.fte >= 0) break;
    }
  }
  console.log(`Replays (${files.length} files): added ${replayFrames} frames`);
}

const n = actList.length;
const obsBuf = Buffer.allocUnsafe(n * 30 * 4);
const actBuf = Buffer.allocUnsafe(n * 4);
let idleN = 0, grapN = 0, heavyN = 0;
for (let i = 0; i < n; i++) {
  const o = obsList[i];
  for (let j = 0; j < 30; j++) obsBuf.writeFloatLE(o[j], (i * 30 + j) * 4);
  actBuf.writeInt32LE(actList[i], i * 4);
  const bits = ACTION_TABLE[actList[i]];
  if (bits === 0) idleN++;
  if (bits & GRAPPLE) grapN++;
  if (bits & HEAVY) heavyN++;
}
fs.writeFileSync('data/bc_obs.bin', obsBuf);
fs.writeFileSync('data/bc_act.bin', actBuf);
console.log(`TOTAL BC DATASET: ${n} frames (idle ${(100*idleN/n).toFixed(1)}%, grapple ${(100*grapN/n).toFixed(1)}%, heavy ${(100*heavyN/n).toFixed(1)}%)`);
