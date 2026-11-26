// train/bridge.mjs
import readline from 'node:readline';
import { WdbEngine } from '../sim/wdb_engine.mjs';

const engine = new WdbEngine();
const send = (obj) => process.stdout.write(JSON.stringify(obj) + '\n');

const rl = readline.createInterface({ input: process.stdin, terminal: false });
let queue = Promise.resolve();

rl.on('line', (line) => {
  queue = queue.then(() => handle(line));
});

async function handle(line) {
  const trimmed = line.trim();
  if (!trimmed) return;
  let req;
  try {
    req = JSON.parse(trimmed);
  } catch {
    return send({ ok: false, error: 'invalid JSON' });
  }
  try {
    if (req.cmd === 'reset') {
      const state = engine.reset(req.players, req.seed ?? 0, req.overrides ?? null);
      send({ ok: true, state });
    } else if (req.cmd === 'step') {
      const res = engine.step(req.actions ?? {}, req.frames ?? 3);
      send({ ok: true, ...res });
    } else if (req.cmd === 'close') {
      send({ ok: true });
      process.exit(0);
    } else {
      send({ ok: false, error: `unknown cmd: ${req.cmd}` });
    }
  } catch (err) {
    send({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
}
