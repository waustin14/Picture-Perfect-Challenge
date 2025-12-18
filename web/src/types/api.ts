export type GameStatus = "lobby" | "active" | "complete";
export type RoundStatus = "pending" | "running" | "ended" | "scoring" | "complete";

export interface GameSummary {
  id: string;
  code: string;
  name: string;
  status: GameStatus;
  created_at: string;
  admin_name: string;
}

export interface PlayerInfo {
  id: string;
  nickname: string;
  joined_at: string;
}

export interface RoundInfo {
  id: string;
  game_id: string;
  round_number: number;
  reference_image_id: string;
  status: RoundStatus;
  duration_seconds: number;
  starts_at: string | null;
  ends_at: string | null;
}

export interface GameState {
  game: GameSummary;
  players: PlayerInfo[];
  rounds: RoundInfo[];
  active_round_id: string | null;
}

export interface GenerateImageRequest {
  player_id: string;
  prompt: string;
  negative_prompt?: string;
}

export interface GenerateImageResponse {
  image_id: string;
  player_id: string;
  round_id: string;
  prompt: string;
  created_at: string;
  image_url: string | null;
}

export interface ReferenceImageInfo {
  name: string;
  url: string;
  size?: number;
  last_modified?: string;
}

export interface ReferenceImagesListResponse {
  images: ReferenceImageInfo[];
}

export interface LeaderboardEntry {
  player_id: string;
  nickname: string;
  total_score: number;
}

export interface LeaderboardResponse {
  game_id: string;
  leaderboard: LeaderboardEntry[];
}
