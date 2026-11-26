import fs from 'node:fs';
import path from 'node:path';
import { bonk, findBallIdx, PPM } from '../sim/wdb_engine.mjs';
import { buildObs, bitsToActionIndex, mirrorBits, TEAM_RED, TEAM_BLUE } from '../sim/features.mjs';

const REPLAY_DIR = process.argv[2] || '/content/wdb-bot/reference/replays';
const files = fs.readdirSync(REPLAY_DIR).filter((f) => f.endsWith('.json')).sort();

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
      down: !!(f & 8), action: !!(f & 16), action2: !!(f & 32),
      bits: f & 63,
    };
  }
  return byTick;
}

function toWdbState(s, ballIdx) {
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
  return {
    ball: bb ? { x: bb.p[0] * PPM, y: bb.p[1] * PPM, xv: bb.lv[0] * PPM, yv: bb.lv[1] * PPM } : { x: 365, y: 250, xv: 0, yv: 0 },
    discs,
    ftu: Math.max(0, s.ftu ?? 0),
  };
}

const obsList = [];
const actList = [];

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

    if (s.ftu === -1 && prevFte === -1 && t % 3 === 0) {
      const ws = toWdbState(s, ballIdx);
      for (let slot = 0; slot < s.discs.length; slot++) {
        const d = s.discs[slot];
        const act = held[slot];
        if (!d || d.ds !== 0 || !act || (d.team !== TEAM_RED && d.team !== TEAM_BLUE)) continue;

        // Pick closest alive opponent to ball
        let oppId = null, bestDist = Infinity;
        for (let j = 0; j < s.discs.length; j++) {
          const o = s.discs[j];
          if (!o || o.ds !== 0 || o.team === d.team || (o.team !== TEAM_RED && o.team !== TEAM_BLUE)) continue;
          const dist = Math.hypot(o.x * PPM - ws.ball.x, o.y * PPM - ws.ball.y);
          if (dist < bestDist) { bestDist = dist; oppId = j; }
        }

        const mirror = d.team === TEAM_RED;
        const obs = buildObs(ws, slot, oppId, mirror);
        const canonBits = mirror ? mirrorBits(act.bits) : act.bits;
        obsList.push(obs);
        actList.push(bitsToActionIndex(canonBits));
      }
    }

    s = bonk.Physics.prototype.step.call(bonk.Physics, s, held, replay.adminInputs?.[t] || null, 30, replay.gameSettings, 1);
    if (prevFte === -1 && s?.fte >= 0) break;
  }
}

const n = actList.length;
const obsBuf = Buffer.allocUnsafe(n * 30 * 4);
const actBuf = Buffer.allocUnsafe(n * 4);
for (let i = 0; i < n; i++) {
  const o = obsList[i];
  for (let j = 0; j < 30; j++) obsBuf.writeFloatLE(o[j], (i * 30 + j) * 4);
  actBuf.writeInt32LE(actList[i], i * 4);
}
fs.writeFileSync('data/bc_obs.bin', obsBuf);
fs.writeFileSync('data/bc_act.bin', actBuf);
console.log(`Harvested ${n} human WDB decision frames from ${files.length} replays.`);
