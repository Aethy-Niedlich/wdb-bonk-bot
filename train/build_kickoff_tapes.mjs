import fs from 'node:fs';
import path from 'node:path';
import { WdbEngine } from '../sim/wdb_engine.mjs';
import { LEFT, RIGHT, UP, DOWN, HEAVY, GRAPPLE, TEAM_RED, TEAM_BLUE, mirrorBits } from '../sim/features.mjs';

const DIR = process.argv[2] || 'data/kickoffs';
const OUT = process.argv[3] || 'data/kickoff_tapes.json';

const CODE_TO_BIT = {
  KeyA: LEFT, ArrowLeft: LEFT,
  KeyD: RIGHT, ArrowRight: RIGHT,
  KeyW: UP, ArrowUp: UP,
  KeyS: DOWN, ArrowDown: DOWN,
  Space: HEAVY, KeyX: HEAVY, ShiftLeft: HEAVY, ShiftRight: HEAVY,
  KeyT: GRAPPLE, KeyZ: GRAPPLE, KeyY: GRAPPLE,
};

function isPlayerRec(b) {
  if (b[6]) return b[6] === 'disc' && b[5] !== null && Math.abs(b[5] - 1) < 0.2;
  return b[0] !== 4 && b[0] !== 6 && b[5] !== null && Math.abs(b[5] - 1) < 0.2;
}

const files = fs.readdirSync(DIR).filter((f) => f.endsWith('.jsonl')).sort();
const tapes = [];
const engine = new WdbEngine();

for (const file of files) {
  const lines = fs.readFileSync(path.join(DIR, file), 'utf8').trim().split('\n');
  const ticks = [], keyEvents = [];
  let meta = null;
  for (const line of lines) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    if (obj.type === 'meta') meta = obj;
    else if (obj.b) ticks.push(obj);
    else if (obj.k && !obj.bot) keyEvents.push(obj);
  }
  if (!ticks.length || !keyEvents.length) continue;

  const b0 = ticks[0].b.find((x) => (x[9] === 1 || (x[0] === 4 && (!x[6] || x[6] !== 'disc'))));
  if (!b0 || Math.hypot(b0[1] * 12 - 365, b0[2] * 12 - 445) > 25) {
    console.log(`SKIP ${file}: did not start at center kickoff`);
    continue;
  }

  let meId = meta?.defence?.me ?? null;
  if (meId == null) {
    const byN = new Map(ticks.map((t) => [t.n, t]));
    const votes = {};
    for (const ke of keyEvents) {
      if (ke.k !== 'down') continue;
      const dir = (ke.code === 'KeyA' || ke.code === 'ArrowLeft') ? -1 : (ke.code === 'KeyD' || ke.code === 'ArrowRight') ? 1 : 0;
      if (!dir) continue;
      const t0 = byN.get(ke.n), t1 = byN.get(ke.n + 4);
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
      if (best && best.gain > 0.8 && best.gain > second + 0.5) votes[best.id] = (votes[best.id] || 0) + 1;
    }
    const ranked = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    if (ranked.length) meId = Number(ranked[0][0]);
  }
  if (meId == null) {
    const counts = new Map();
    for (const tk of ticks) {
      for (const b of tk.b) if (isPlayerRec(b)) counts.set(b[0], (counts.get(b[0]) || 0) + 1);
    }
    const ids = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    if (ids.length) meId = ids[0][0];
  }

  const held = new Set();
  let ki = 0;
  let side = null;
  const rawBits = [];

  for (const tk of ticks) {
    while (ki < keyEvents.length && keyEvents[ki].n < tk.n) {
      const ke = keyEvents[ki++];
      if (ke.k === 'sync' && Array.isArray(ke.held)) {
        held.clear();
        for (const c of ke.held) if (CODE_TO_BIT[c]) held.add(c);
      } else if (CODE_TO_BIT[ke.code]) {
        if (ke.k === 'down') held.add(ke.code);
        else held.delete(ke.code);
      }
    }
    const me = tk.b.find((x) => x[0] === meId && (!x[6] || x[6] === 'disc'));
    if (!me) continue;
    if (side === null) side = me[1] < 30.417 ? 'blue' : 'red';
    let b = 0;
    for (const c of held) b |= (CODE_TO_BIT[c] || 0);
    rawBits.push(b);
  }

  const firstActive = rawBits.findIndex((b) => b !== 0);
  if (firstActive < 0) continue;
  const start = Math.max(0, firstActive - 1);
  const mirror = side === 'red';
  const canon = rawBits.slice(start, start + 600).map((b) => (mirror ? mirrorBits(b) : b));

  // Verify full shot inside bonk-core.mjs
  engine.reset([{ id: 0, team: TEAM_RED }, { id: 1, team: TEAM_BLUE }], 1, { skipCountdown: true });
  let maxVx = 0, maxX = 365, oppDied = false, shootLen = canon.length;
  for (let i = 0; i < canon.length; i++) {
    const res = engine.step({ '1': canon[i], '0': 0 }, 1);
    if (res.state.ball.xv > maxVx) maxVx = res.state.ball.xv;
    if (res.state.ball.x > maxX) maxX = res.state.ball.x;
    if (res.winner === TEAM_BLUE) { oppDied = true; shootLen = i + 1; break; }
    if (res.winner !== null) break;
  }
  const keep = oppDied;
  console.log(`${keep ? 'KEEP' : 'SKIP'} ${file}: side=${side}, len=${shootLen}, ballMaxX=${Math.round(maxX)}, maxVx=${Math.round(maxVx)}, kill=${oppDied}`);
  if (keep) tapes.push({ file, canon: canon.slice(0, shootLen) });
}

fs.writeFileSync(OUT, JSON.stringify({ tapes }, null, 1));
console.log(`Saved ${tapes.length} verified lethal kickoff tapes -> ${OUT}`);
