import 'dotenv/config';
import {
  createRoom, joinRoom, decodeInitialState, parseInputFrame,
  TEAM_BLUE, TEAM_RED, type BonkRoom,
} from 'bonktools';
import { WdbEngine } from '../sim/wdb_engine.mjs';
import { ACTION_TABLE, buildObs, mirrorBits, type WdbState } from './features.js';
import { WdbPolicy } from './policy.js';

const POLICY_PATH = process.env.WDB_MODEL ?? 'models/wdb.policy.json';
const ROOM_URL = process.env.BONK_ROOM_URL;
const LAG_FRAMES = 4;
const KEEP_HISTORY = 45;

const policy = WdbPolicy.fromFile(POLICY_PATH);
const engine = new WdbEngine();

async function connect(): Promise<BonkRoom> {
  const user = process.env.BONK_USER;
  const pass = process.env.BONK_PASS;
  const auth = user && pass
    ? { type: 'registered' as const, username: user, password: pass }
    : { type: 'guest' as const, guestName: process.env.BONK_GUEST_NAME ?? 'WDB_Bot' };

  if (ROOM_URL) {
    console.log('Joining practice room:', ROOM_URL);
    return joinRoom(ROOM_URL, { auth });
  }
  if (auth.type !== 'registered') {
    throw new Error('Creating a room requires BONK_USER & BONK_PASS, or set BONK_ROOM_URL to join yours.');
  }
  const room = await createRoom({
    auth,
    desiredState: {
      roomName: 'WDB Practice Bot',
      password: process.env.BONK_ROOM_PASS ?? 'wdb123',
      maxPlayers: 4,
      mode: 'sp',
      rounds: 5,
    },
    hidden: true,
  });
  console.log('Room link:', room.shareLink);
  return room;
}

async function main() {
  const room = await connect();
  room.enableAntiAfk();

  let active = false;
  let startedAtMs = 0;
  let simFrame = 0;
  let lastDecisionFrame = -999;
  let lastSentBits = 0;
  let prevFtu = 90;
  let winLimit = 3;
  let heldActions: Record<string, number> = {};
  let inputLog = new Map<number, Record<string, number>>();
  let history = new Map<number, { rawState: unknown; held: Record<string, number>; ftu: number }>();
  let seenOppLog = false;
      let meSentLogs = 0;
      let rolledBackState: any = null;
  let loop: NodeJS.Timeout | null = null;

  const stopMatch = (reason?: string) => {
    if (active && reason) console.log('[bot] match stopped:', reason);
    active = false;
    if (loop) { clearInterval(loop); loop = null; }
    if (lastSentBits !== 0) { room.sendInput(0); lastSentBits = 0; }
  };

  if (room.state.myId !== null) {
    const me = room.state.players.get(room.state.myId);
    if (me && me.team !== TEAM_BLUE && me.team !== TEAM_RED) room.joinTeam(TEAM_BLUE);
  }

  const stepOneFrame = (f: number): WdbState => {
    const atF = inputLog.get(f);
    if (atF) Object.assign(heldActions, atF);
    const res = engine.step(heldActions, 1);
    history.set(f + 1, {
      rawState: engine.getRawState(),
      held: { ...heldActions },
      ftu: res.state.ftu,
    });
    const cutoff = f - KEEP_HISTORY;
    if (cutoff > 0) {
      history.delete(cutoff);
      inputLog.delete(cutoff);
    }
    return res.state;
  };

  const queueInput = (playerId: number, bits: number, f: number, src?: string) => {
    if (!active) return;
    if (src && playerId !== room.state.myId) {
      if (!seenOppLog) seenOppLog = true;
      console.log(`[opp-${src}]`, { bits, f, simFrame, diff: f - simFrame });
    }
    let bucket = inputLog.get(f);
    if (!bucket) { bucket = {}; inputLog.set(f, bucket); }
    bucket[String(playerId)] = bits;

    // Rollback & replay if an opponent packet arrived for a frame we already stepped
    if (f < simFrame && history.has(f)) {
          console.log('[rollback]', { f, simFrame, lag: simFrame - f });
          const snap = history.get(f)!;
          engine.setRawState(snap.rawState);
          (engine as any).lastBits = [];
          heldActions = { ...snap.held };
          prevFtu = snap.ftu;
          const target = simFrame;
          rolledBackState = null;
          for (let t = f; t < target; t++) {
            if (t > f && rolledBackState) prevFtu = rolledBackState.ftu;
            rolledBackState = stepOneFrame(t);
          }
        } else if (f < simFrame) {
          console.log('[rollback-miss]', { f, simFrame, lag: simFrame - f });
        } else if (f < simFrame) {
      heldActions[String(playerId)] = bits;
    }
  };

  room.on('peer-input', (ev) => {
    if (ev.playerId === room.state.myId) return;
    const parsed = parseInputFrame(ev.data);
    if (!parsed) return;
    const nowF = Math.floor((Date.now() - startedAtMs) * 30 / 1000);
    const f = Math.max(0, parsed.f + Math.round((nowF - parsed.f) / 65536) * 65536);
    queueInput(ev.playerId, parsed.i, f, 'webrtc');
  });

  room.on('raw-packet', (pkt: unknown) => {
    const p = pkt as { type?: string; raw?: unknown[] };
    if (!p || p.type !== 'UNKNOWN' || !Array.isArray(p.raw)) return;
    const [id, playerId, data] = p.raw;
    if (id === 7 && typeof playerId === 'number' && playerId !== room.state.myId && data && typeof (data as { i?: number; f?: number }).i === 'number') {
      const d = data as { i: number; f: number };
      queueInput(playerId, d.i, d.f, 'pkt7');
    } else if (id !== 1 && id !== 7) {
      console.log('[raw-other]', p.raw);
    }
  });

  room.on('game-start', (pkt) => {
    stopMatch();
    startedAtMs = Date.now();
    (room as any).gameStartedAtMs = startedAtMs;

    const myId = room.state.myId;
    if (myId === null) return;

    const lineup: Array<{ id: number; team: number } | null> = [];
    const activePlayers: Array<{ id: number; team: number }> = [];
    for (const p of room.state.players.values()) {
      if (p.team === TEAM_RED || p.team === TEAM_BLUE) {
        lineup[p.id] = { id: p.id, team: p.team };
        activePlayers.push({ id: p.id, team: p.team });
      }
    }
    const me = activePlayers.find((p) => p.id === myId);
    const opp = activePlayers.find((p) => p.id !== myId && p.team !== me?.team);
    if (!me || !opp) return;

    let decodedIS: unknown = null;
    if (pkt?.is) {
      try {
        decodedIS = decodeInitialState(pkt.is);
        console.log('[bot] decoded live GAME_START initialState (zero-desync)');
      } catch (err) {
        console.warn('[bot] decodeInitialState fallback:', err);
      }
    }
    const initState = engine.reset(lineup, 0, {
      initialState: decodedIS ?? undefined,
      gs: pkt?.gs as Record<string, unknown>,
    });

    const gs = pkt?.gs as { wl?: number } | undefined;
    winLimit = typeof gs?.wl === 'number' && gs.wl > 0 ? gs.wl : 3;

    simFrame = 0;
    lastDecisionFrame = -999;
    lastSentBits = -1;
    prevFtu = initState.ftu;
    heldActions = {};
    inputLog = new Map();
    history = new Map([[0, { rawState: engine.getRawState(), held: {}, ftu: initState.ftu }]]);
    seenOppLog = false;
    active = true;

    console.log('[bot] match started', { me, opp, winLimit, ftu: initState.ftu });

    loop = setInterval(() => {
      if (!active) return;
      const nowFrame = Math.floor((Date.now() - startedAtMs) * 30 / 1000);
      const targetFrame = Math.max(0, nowFrame - LAG_FRAMES);
      let latestState: WdbState = engine.getState();

      if (rolledBackState) {
          latestState = rolledBackState;
          rolledBackState = null;
        }
        while (simFrame < targetFrame) {
        latestState = stepOneFrame(simFrame);
        simFrame++;

        const sc = latestState.scores ?? [];
        const maxScore = Math.max(Number(sc[TEAM_RED] ?? 0), Number(sc[TEAM_BLUE] ?? 0));
        if (prevFtu <= 0 && latestState.ftu > 0) {
          if (maxScore >= winLimit) {
            stopMatch(`win limit reached (${maxScore}/${winLimit})`);
            return;
          }
        }
        prevFtu = latestState.ftu;
      }

      if (simFrame > lastDecisionFrame) {
        lastDecisionFrame = simFrame;
        const d = latestState.discs[String(me.id)];
        let bits = 0;
        if (d && d.alive && latestState.ftu <= 0) {
          const mirror = me.team === TEAM_RED;
          const obs = buildObs(latestState, me.id, opp.id, mirror);
          const rawBits = ACTION_TABLE[policy.act(obs)] ?? 0;
          bits = mirror ? mirrorBits(rawBits) : rawBits;
        }
        if (bits !== lastSentBits) {
        lastSentBits = bits;
        const sent = room.sendInput(bits);
        if (meSentLogs < 5) {
          meSentLogs++;
          console.log('[me-sent]', { bits, f: sent.f, nowFrame, simFrame });
        }
        queueInput(me.id, bits, sent.f);
      }
        if (simFrame % 30 === 0) {
          const o = latestState.discs[String(opp.id)];
          const fmt = (p?: { x: number; y: number; alive: boolean }) =>
            p?.alive ? `(${Math.round(p.x)},${Math.round(p.y)})` : 'DEAD';
          console.log(
            `[f=${simFrame}] me=${fmt(d)} opp=${fmt(o)}`,
            `ball=(${Math.round(latestState.ball.x)},${Math.round(latestState.ball.y)})`,
            `hold=${!!d?.holdingBall} bits=${bits}`,
          );
        }
      }
    }, 33);
  });

  room.on('game-end', () => stopMatch('game-end'));
  room.on('ready-change', () => stopMatch('ready-change (back in lobby)'));
  room.on('all-ready-reset', () => stopMatch('all-ready-reset'));
  room.on('map-switch', () => stopMatch('map-switch'));
  room.on('gamemode-change', () => stopMatch('gamemode-change'));
  room.on('change-rounds', () => stopMatch('change-rounds'));
  room.on('teamlock-toggle', () => stopMatch('teamlock-toggle'));
  room.on('host-leave', () => stopMatch('host-leave'));
  room.on('room-dead', (r) => stopMatch(`room-dead: ${String(r)}`));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
