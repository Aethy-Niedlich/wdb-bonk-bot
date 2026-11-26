import type { WdbState } from './features.js';
export class WdbEngine {
  reset(players?: Array<{ id: number; team: number } | null>, seed?: number, opts?: { initialState?: unknown; gs?: Record<string, unknown>; skipCountdown?: boolean } | null): WdbState;
  step(actions?: Record<string, number>, frames?: number): { state: WdbState; frame: number; winner: number | null };
  getRawState(): unknown;
  setRawState(rawState: unknown): void;
  getState(): WdbState;
}
