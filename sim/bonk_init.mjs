import { createRequire } from 'node:module';
import fs from 'node:fs';

const require = createRequire(import.meta.url);

function ensureBonkGlobals() {
  const g = globalThis;
  if (!g.window) g.window = g;
  if (!g.self) g.self = g;
  if (!g.document) {
    g.document = {
      createElement: () => ({ getContext: () => null, style: {}, width: 0, height: 0 }),
      body: { appendChild() {} },
      getElementById: () => null,
    };
  }
  const rdr = { width: 0, height: 0, resize() {}, render() {}, destroy() {}, view: {}, stage: { children: [] }, forceRendererSize() {}, backgroundColor: 0 };
  g.PIXI ??= {
    autoDetectRenderer: () => ({ ...rdr }),
    settings: {}, utils: {},
    Container: class { addChild() {} },
    Graphics: class { beginFill() {} lineStyle() {} drawRect() {} beginTextureFill() {} },
    Text: class { constructor() { this.resolution = 1; this.width = 100; this.height = 20; this.x = 0; this.y = 0; this.alpha = 1; this.visible = true; } },
    filters: { AdjustmentFilter: class {}, BlurFilter: class {}, FXAAFilter: class {} },
    resources: { SVGResource: class { load() { return Promise.resolve(); } } },
  };
  g.peerjs ??= { peerjs: { Peer: class {} } };
  g.AudioContext ??= class { constructor() { this.destination = {}; } createGain() { return { connect() {}, gain: { value: 0 } }; } };
  g.Howl ??= class { constructor() {} play() {} volume() { return this; } stereo() { return this; } on() {} };
  g.GameResources ??= { soundStrings: { capIncrease: '', capDecrease: '', discDisc: '', platBounce: '' }, svgStrings: {} };
  g.dcodeIO ??= { PSON: { StaticPair: class { constructor(d) { this.dict = d; } encode() { return { toBase64: () => '' }; } decode() { return {}; } } } };

  if (!g.SafeTrig) {
    const trigSrc = fs.readFileSync(new URL('./bonk/SafeTrig.js', import.meta.url), 'utf8');
    (0, eval)(trigSrc + '\n;globalThis.SafeTrig = SafeTrig;');
  }
}

ensureBonkGlobals();

let amd = null;
const prevDefine = globalThis.define;
globalThis.define = (...a) => {
  const f = a[a.length - 1];
  if (typeof f !== 'function') { amd = f; return; }
  const deps = a.find((x) => Array.isArray(x)) || [];
  const ex = {}, mo = { exports: ex };
  let v = deps.map((d) => (d === 'exports' ? ex : d === 'module' ? mo : globalThis[d]));
  if (!deps.length && f.length) v = [ex, mo].slice(0, f.length);
  const r = f(...v);
  amd = r != null ? r : Object.keys(ex).length ? ex : mo.exports;
};
globalThis.define.amd = true;

let box2d = require('./bonk/Box2DModuleGlMod.js');
const isB2 = (x) => x && typeof x === 'object' && (x.Common || x.Dynamics);
box2d = isB2(box2d) ? box2d : isB2(globalThis.Box2D) ? globalThis.Box2D : amd;
if (prevDefine !== undefined) globalThis.define = prevDefine;

export { box2d };
