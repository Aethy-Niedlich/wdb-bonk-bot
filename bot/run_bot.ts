// bot/run_bot.ts
import 'dotenv/config';
import { createRoom, joinRoom, TEAM_BLUE, TEAM_RED, type BonkRoom } from 'bonktools';
import { WdbEngine } from '../sim/wdb_engine.mjs';
import { ACTION_TABLE, buildObs, mirrorBits, type WdbState } from './features.js';
import { WdbPolicy } from './policy.js';

const POLICY_PATH = process.env.WDB_MODEL ?? 'models/wdb.policy.json';
const ROOM_URL = process.env.BONK_ROOM_URL;
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
  console.log('Creating private Grapple (sp) room...');
  const room = await createRoom({
    auth,
    desiredState: {
      roomName: 'WDB Practice Bot',
      password: process.env.BONK_ROOM_PASS ?? 'wdb123',
      maxPlayers: 4,
      mode: 'sp', // Grapple mode
      rounds: 5,
    },
    hidden: true,
  });
  console.log('Room link:', room.shareLink);
  return room;
}

async function main() {
  const room = await connect();
  // Enable WebRTC peer-input listener
  room.enableAntiAfk();

  let active = false;
  let lastSentBits = -1;
  let liveActions: Record<string, number> = {};
  let loop: NodeJS.Timeout | null = null;

  const stopMatch = () => {
    active = false;
    if (loop) {
      clearInterval(loop);
      loop = null;
    }
    if (lastSentBits !== 0) {
      room.sendInput(0); // Release held keys cleanly
      lastSentBits = 0;
    }
  };

  // When joining a room, put bot on Blue if currently Spectator
  if (room.state.myId !== null) {
    const me = room.state.players.get(room.state.myId);
    if (me && me.team !== TEAM_BLUE && me.team !== TEAM_RED) {
      room.joinTeam(TEAM_BLUE);
    }
  }

  // Receive opponent input frames (WebRTC peer-input or Socket.IO packet 7)
  room.on('peer-input', ({ playerId, data }) => {
    const d = data as { i?: number } | null;
    if (d && typeof d.i === 'number') {
      liveActions[String(playerId)] = d.i;
    }
  });

  room.on('raw-packet', (pkt: unknown) => {
    // Fallback for Socket.IO input packet [7, playerId, { i, f, c }]
    if (Array.isArray(pkt) && pkt[0] === 7 && typeof pkt[1] === 'number' && pkt[2] && typeof pkt[2].i === 'number') {
      liveActions[String(pkt[1])] = pkt[2].i;
    }
  });

  room.on('game-start', () => {
    stopMatch();
    const myId = room.state.myId;
    if (myId === null) return;

    const lineup: Array<{ id: number; team: number }> = [];
    for (const p of room.state.players.values()) {
      if (p.team === TEAM_RED || p.team === TEAM_BLUE) {
        lineup.push({ id: p.id, team: p.team });
      }
    }
    const me = lineup.find((p) => p.id === myId);
    const opp = lineup.find((p) => p.id !== myId && p.team !== me?.team);
    if (!me || !opp) return;

    liveActions = {};
    engine.reset(lineup, 0, { ftu: 90 }); // ~3s countdown
    active = true;
    lastSentBits = -1;

    // Decision loop at 100 ms (3 frames at 30 Hz — matches training cadence)
    loop = setInterval(() => {
      if (!active) return;
      const { state, winner } = engine.step(liveActions, 3);
      if (winner !== null) return;

      const mirror = me.team === TEAM_BLUE;
      const obs = buildObs(state as WdbState, me.id, opp.id, mirror);
      const rawBits = ACTION_TABLE[policy.act(obs)] ?? 0;
      const bits = mirror ? mirrorBits(rawBits) : rawBits;

      // Track our own input locally (bonk.io does not echo host's own inputs)
      liveActions[String(me.id)] = bits;

      // Only send network packets when key state actually changes
      if (bits !== lastSentBits) {
        lastSentBits = bits;
        room.sendInput(bits);
      }
    }, 100);
  });

  room.on('game-end', stopMatch);
  room.on('room-dead', (r) => {
    stopMatch();
    console.error('Room closed:', r);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
