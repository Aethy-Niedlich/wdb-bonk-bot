// bot/policy.ts
import { readFileSync } from 'node:fs';
import { ACTION_TABLE, OBS_SIZE } from './features.js';

interface LayerJson { in: number; out: number; w: string; b: string; }
interface PolicyJson {
  format: string;
  obsSize: number;
  actionTable: number[];
  activation: string;
  layers: LayerJson[];
}

function decodeF32(b64: string, expected: number): Float32Array {
  const buf = Buffer.from(b64, 'base64');
  const out = new Float32Array(expected);
  for (let i = 0; i < expected; i++) out[i] = buf.readFloatLE(i * 4);
  return out;
}

export class WdbPolicy {
  private readonly layers: Array<{ inSize: number; outSize: number; w: Float32Array; b: Float32Array }>;

  constructor(json: PolicyJson) {
    if (json.format !== 'wdb-rl-policy/1') throw new Error(`Bad format: ${json.format}`);
    if (json.obsSize !== OBS_SIZE) throw new Error(`Expected obsSize ${OBS_SIZE}, got ${json.obsSize}`);
    if (json.actionTable.length !== ACTION_TABLE.length || json.actionTable.some((v, i) => v !== ACTION_TABLE[i])) {
      throw new Error('ACTION_TABLE mismatch between policy.json and features.ts');
    }
    this.layers = json.layers.map((l) => ({
      inSize: l.in,
      outSize: l.out,
      w: decodeF32(l.w, l.in * l.out),
      b: decodeF32(l.b, l.out),
    }));
  }

  static fromFile(path: string): WdbPolicy {
    return new WdbPolicy(JSON.parse(readFileSync(path, 'utf8')) as PolicyJson);
  }

  act(obs: ArrayLike<number>): number {
    let x: ArrayLike<number> = obs;
    for (let li = 0; li < this.layers.length; li++) {
      const { inSize, outSize, w, b } = this.layers[li]!;
      const y = new Float64Array(outSize);
      for (let o = 0; o < outSize; o++) {
        let sum = b[o]!;
        const row = o * inSize;
        for (let i = 0; i < inSize; i++) sum += w[row + i]! * x[i]!;
        y[o] = li < this.layers.length - 1 ? Math.tanh(sum) : sum;
      }
      x = y;
    }
    let best = 0;
    for (let i = 1; i < x.length; i++) if (x[i]! > x[best]!) best = i;
    return best;
  }
}
