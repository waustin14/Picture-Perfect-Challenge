import { useEffect, useRef } from "react";
import { wsUrl } from "../api/client";

export interface PlayerScoreInfo {
  player_id: string;
  nickname: string;
  image_id: string | null;
  image_url: string | null;
  similarity_score: number;
  score: number;
}

export type GameEvent =
  | { type: "round_started"; round_id: string; round_number: number; duration_seconds: number; starts_at: string | null; ends_at: string | null; reference_image_url: string }
  | { type: "round_tick"; round_id: string; round_number: number; remaining_seconds: number }
  | { type: "round_ended"; round_id: string; round_number: number; ended_by?: string; status?: string }
  | { type: "image_attempt_created"; round_id: string; player_id: string; image_id: string; prompt: string; created_at: string; image_url?: string | null }
  | { type: "image_submitted"; round_id: string; player_id: string; image_id: string }
  | { type: "image_unsubmitted"; round_id: string; player_id: string; image_id: string }
  | { type: "round_results"; round_id: string; round_number: number; reference_image_url: string; status: string; player_scores: PlayerScoreInfo[]; game_status: string }
  | { type: "round_scored"; round_id: string; scores: PlayerScoreInfo[] };

export function useGameWebSocket(gameId: string, onEvent: (event: GameEvent) => void) {
  const eventRef = useRef(onEvent);
  eventRef.current = onEvent;

  useEffect(() => {
    if (!gameId) return;

    const socket = new WebSocket(wsUrl(`/ws/games/${gameId}`));

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        eventRef.current(data);
      } catch (e) {
        console.error("WS parse error", e);
      }
    };

    socket.onopen = () => {
      console.log("WebSocket connected");
    };

    socket.onclose = () => {
      console.log("WebSocket closed");
    };

    // We never send messages from client for now, but you could if needed.
    return () => socket.close();
  }, [gameId]);
}
