export const WIDTH: number;
export const HEIGHT: number;
export const HALF_W: number;
export const HALF_H: number;
export const TEAM_RED: number;
export const TEAM_BLUE: number;
export const ACTION_TABLE: readonly number[];
export const OBS_SIZE: number;
export function mirrorBits(bits: number): number;
export function bitsToActionIndex(bits: number): number;
export interface DiscState {
  x: number; y: number; xv: number; yv: number;
  team: number; alive: boolean; heavy: boolean;
  stamina: number; grappleMs: number; onFloor: boolean; holdingBall: boolean;
}
export interface WdbState {
  ball: { x: number; y: number; xv: number; yv: number };
  discs: Record<string, DiscState>;
  ftu: number;
  fte?: number;
  scores?: Array<number | null>;
}
export function buildObs(state: WdbState, selfId: number, oppId: number | null, mirror: boolean): Float32Array;
