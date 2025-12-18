import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { apiGet, apiPost, rewriteMinioUrl } from "../api/client";
import type { GameState, RoundInfo, LeaderboardEntry, LeaderboardResponse } from "../types/api";
import { useGameWebSocket } from "../hooks/useGameWebSocket";
import type { GameEvent, PlayerScoreInfo } from "../hooks/useGameWebSocket";
import { TimerDisplay } from "../components/TimerDisplay";

interface RoundResults {
  round_number: number;
  reference_image_url: string;
  player_scores: PlayerScoreInfo[];
}

interface Attempt {
  id: string;
  prompt: string;
  playerId: string;
  createdAt: string;
  isSubmission: boolean;
  imageUrl?: string | null;
}

export function PlayerGamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null);
  const [roundResults, setRoundResults] = useState<RoundResults | null>(null);

  const refreshGameState = useCallback(() => {
    if (!gameId) return;
    apiGet<GameState>(`/games/${gameId}/state`)
      .then(setGameState)
      .catch(console.error);
  }, [gameId]);

  const refreshLeaderboard = useCallback(() => {
    if (!gameId) return;
    apiGet<LeaderboardResponse>(`/games/${gameId}/leaderboard`)
      .then((res) => setLeaderboard(res.leaderboard))
      .catch(console.error);
  }, [gameId]);

  const playerId = localStorage.getItem("playerId") || "";
  const nickname = localStorage.getItem("playerNickname") || "";

  useEffect(() => {
    if (!gameId) return;
    refreshGameState();
    refreshLeaderboard();
  }, [refreshGameState, refreshLeaderboard]);

  const activeRound: RoundInfo | undefined = useMemo(() => {
    if (!gameState || !gameState.active_round_id) return undefined;
    return gameState.rounds.find((r) => r.id === gameState.active_round_id);
  }, [gameState]);

  useGameWebSocket(gameId ?? "", (event: GameEvent) => {
    if (event.type === "round_started") {
      setRemainingSeconds(event.duration_seconds);
      setAttempts([]);
      setSelectedImageId(null);
      setReferenceImageUrl(event.reference_image_url);
      setRoundResults(null);
      refreshGameState();
    } else if (event.type === "round_tick") {
      setRemainingSeconds(event.remaining_seconds);
    } else if (event.type === "round_ended") {
      setRemainingSeconds(0);
      refreshGameState();
    } else if (event.type === "image_attempt_created") {
      if (event.player_id === playerId) {
        setAttempts((prev) => [
          ...prev,
          {
            id: event.image_id,
            prompt: event.prompt,
            playerId: event.player_id,
            createdAt: event.created_at,
            isSubmission: false,
            imageUrl: event.image_url ?? null,
          },
        ]);
      }
    } else if (event.type === "image_submitted") {
      if (event.player_id === playerId) {
        setAttempts((prev) =>
          prev.map((a) =>
            a.id === event.image_id ? { ...a, isSubmission: true } : a
          )
        );
        setSelectedImageId(null);
      }
    } else if (event.type === "image_unsubmitted") {
      if (event.player_id === playerId) {
        setAttempts((prev) =>
          prev.map((a) =>
            a.id === event.image_id ? { ...a, isSubmission: false } : a
          )
        );
      }
    } else if (event.type === "round_results") {
      setRoundResults({
        round_number: event.round_number,
        reference_image_url: event.reference_image_url,
        player_scores: event.player_scores,
      });
      refreshLeaderboard();
      refreshGameState();
    }
  });

  const handleGenerate = async () => {
    if (!activeRound || activeRound.status !== "running" || !prompt.trim() || !playerId) return;
    setGenerating(true);
    try {
      const payload: { player_id: string; prompt: string; negative_prompt?: string } = {
        player_id: playerId,
        prompt,
      };
      if (negativePrompt.trim()) {
        payload.negative_prompt = negativePrompt;
      }
      await apiPost("/rounds/" + activeRound.id + "/generate", payload);
      setPrompt("");
      setNegativePrompt("");
    } catch (e) {
      console.error(e);
    } finally {
      setGenerating(false);
    }
  };

  const handleSubmitImage = async () => {
    if (!activeRound || !selectedImageId || !playerId) return;
    setSubmitting(true);
    try {
      await apiPost("/rounds/" + activeRound.id + "/submit", {
        player_id: playerId,
        image_id: selectedImageId,
      });
    } catch (e) {
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  const maxAttemptsReached = attempts.length >= 6;
  const roundEnded = remainingSeconds !== null && remainingSeconds <= 0;
  const roundRunning = activeRound?.status === "running";
  const hasSubmitted = attempts.some(a => a.isSubmission);

  const playerScore = useMemo(() => {
    const entry = leaderboard.find((e) => e.player_id === playerId);
    return entry?.total_score ?? 0;
  }, [leaderboard, playerId]);

  const selectedAttempt = useMemo(() => {
    return attempts.find(a => a.id === selectedImageId);
  }, [attempts, selectedImageId]);

  // Render Results Screen
  if (roundResults) {
    const playerIndex = roundResults.player_scores.findIndex(ps => ps.player_id === playerId);
    const playerResult = roundResults.player_scores[playerIndex];

    return (
      <div className="game-container">
        <header className="game-header">
          <div className="game-logo">Picture Perfect</div>
          <div className="player-info">
            <span className="player-name">{nickname}</span>
            <span className="player-score">{playerScore} pts</span>
          </div>
        </header>

        <div className="results-container">
          <h2 className="results-title">Round {roundResults.round_number} Results</h2>

          <div className="reference-section" style={{ padding: "var(--spacing-sm)" }}>
            <div className="reference-label">Reference Image</div>
            <img
              src={rewriteMinioUrl(roundResults.reference_image_url) || ""}
              alt="Reference"
              className="reference-image"
              style={{ maxHeight: "120px" }}
            />
          </div>

          <div className="results-podium">
            {roundResults.player_scores.slice(0, 3).map((ps, index) => {
              const podiumClass = index === 0 ? "podium-entry--gold" :
                                  index === 1 ? "podium-entry--silver" : "podium-entry--bronze";
              return (
                <div key={ps.player_id} className={`podium-entry ${podiumClass}`}>
                  <span className="podium-medal">
                    {index === 0 ? "🥇" : index === 1 ? "🥈" : "🥉"}
                  </span>
                  {ps.image_url ? (
                    <img
                      src={rewriteMinioUrl(ps.image_url) || ""}
                      alt={ps.nickname}
                      className="podium-image"
                    />
                  ) : (
                    <div className="podium-image" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                      No image
                    </div>
                  )}
                  <span className="podium-name">{ps.nickname}</span>
                  <span className="podium-score">{ps.score} pts</span>
                  <span className="podium-match">{Math.round(ps.similarity_score * 100)}% match</span>
                </div>
              );
            })}
          </div>

          {playerIndex >= 3 && playerResult && (
            <div style={{ marginTop: "var(--spacing-lg)", padding: "var(--spacing-md)", background: "var(--bg-card)", borderRadius: "var(--radius-lg)" }}>
              <div className="text-muted text-small">Your Result: #{playerIndex + 1}</div>
              <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--secondary)" }}>{playerResult.score} pts</div>
              <div className="text-muted text-small">{Math.round(playerResult.similarity_score * 100)}% match</div>
            </div>
          )}

          <p className="text-muted">Waiting for the next round...</p>
        </div>
      </div>
    );
  }

  // Render Active Round
  if (activeRound) {
    return (
      <div className="game-container">
        <header className="game-header">
          <div className="game-logo">Picture Perfect</div>
          <div className="player-info">
            <span className="player-name">{nickname}</span>
            <span className="player-score">{playerScore} pts</span>
            <TimerDisplay remainingSeconds={remainingSeconds} />
          </div>
        </header>

        <div className="game-main">
          {/* Sidebar with leaderboard */}
          <aside className="game-sidebar">
            <div className="leaderboard">
              <div className="leaderboard-title">Leaderboard</div>
              <div className="leaderboard-list">
                {leaderboard.length === 0 ? (
                  <div className="text-muted text-small">No scores yet</div>
                ) : (
                  leaderboard.map((entry, index) => (
                    <div
                      key={entry.player_id}
                      className={`leaderboard-entry ${entry.player_id === playerId ? "leaderboard-entry--self" : ""}`}
                    >
                      <span className="leaderboard-rank">
                        {index === 0 ? "🥇" : index === 1 ? "🥈" : index === 2 ? "🥉" : `#${index + 1}`}
                      </span>
                      <span className="leaderboard-name">{entry.nickname}</span>
                      <span className="leaderboard-score">{entry.total_score}</span>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Reference Image */}
            <div className="reference-section">
              <div className="reference-label">Match This Image</div>
              {referenceImageUrl ? (
                <img
                  src={rewriteMinioUrl(referenceImageUrl) || ""}
                  alt="Reference"
                  className="reference-image"
                />
              ) : (
                <div className="text-muted">Loading...</div>
              )}
            </div>
          </aside>

          {/* Main Content */}
          <div className="game-content">
            {/* Preview and Image Selection Row */}
            <div style={{ display: "flex", gap: "var(--spacing-md)", flex: 1, minHeight: 0 }}>
              {/* Preview */}
              <div className="preview-section" style={{ flex: 2 }}>
                <div className="preview-image">
                  {selectedAttempt?.imageUrl ? (
                    <img
                      src={rewriteMinioUrl(selectedAttempt.imageUrl) || ""}
                      alt="Selected"
                    />
                  ) : (
                    <div className="preview-placeholder">
                      {attempts.length === 0
                        ? "Generate an image to get started"
                        : "Select an image to preview"}
                    </div>
                  )}
                </div>
                {selectedAttempt && (
                  <div className="text-muted text-small mt-sm" style={{ textAlign: "center" }}>
                    {selectedAttempt.prompt}
                  </div>
                )}
              </div>

              {/* Generated Images Grid */}
              <div style={{ flex: 1, minWidth: "200px" }}>
                <div className="text-muted text-small mb-sm">
                  Your Images ({attempts.length}/6)
                </div>
                <div className="image-grid">
                  {attempts.map((a) => (
                    <div
                      key={a.id}
                      className={`image-thumb ${a.id === selectedImageId ? "image-thumb--selected" : ""} ${a.isSubmission ? "image-thumb--submitted" : ""}`}
                      onClick={() => !a.isSubmission && setSelectedImageId(a.id)}
                    >
                      {a.imageUrl ? (
                        <img src={rewriteMinioUrl(a.imageUrl) || ""} alt="Generated" />
                      ) : (
                        <div className="image-thumb--loading">...</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Prompt Input and Actions */}
            <div className="prompt-section">
              <div style={{ display: "flex", gap: "var(--spacing-sm)", alignItems: "flex-end" }}>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "var(--spacing-xs)" }}>
                  <input
                    type="text"
                    className="prompt-input"
                    placeholder={maxAttemptsReached ? "Max attempts reached" : "Describe the image you want to generate..."}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleGenerate()}
                    disabled={!roundRunning || roundEnded || maxAttemptsReached}
                  />
                  <input
                    type="text"
                    className="prompt-input"
                    style={{ fontSize: "0.85rem", padding: "var(--spacing-xs) var(--spacing-sm)" }}
                    placeholder="Negative prompt (optional) - things to avoid..."
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleGenerate()}
                    disabled={!roundRunning || roundEnded || maxAttemptsReached}
                  />
                </div>
                <button
                  className="btn btn--primary"
                  onClick={handleGenerate}
                  disabled={!roundRunning || roundEnded || generating || maxAttemptsReached || !prompt.trim()}
                >
                  {generating ? "Generating..." : "Generate"}
                </button>
                <button
                  className="btn btn--secondary"
                  onClick={handleSubmitImage}
                  disabled={!selectedImageId || roundEnded || submitting || hasSubmitted}
                >
                  {hasSubmitted ? "Submitted!" : submitting ? "Submitting..." : "Submit"}
                </button>
              </div>
              {hasSubmitted && (
                <div className="text-small mt-sm" style={{ color: "var(--secondary)" }}>
                  Your image has been submitted. Good luck!
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Render Waiting State
  return (
    <div className="game-container">
      <header className="game-header">
        <div className="game-logo">Picture Perfect</div>
        <div className="player-info">
          <span className="player-name">{nickname}</span>
          <span className="player-score">{playerScore} pts</span>
          <TimerDisplay remainingSeconds={null} />
        </div>
      </header>

      <div className="game-main">
        <aside className="game-sidebar">
          <div className="leaderboard">
            <div className="leaderboard-title">Leaderboard</div>
            <div className="leaderboard-list">
              {leaderboard.length === 0 ? (
                <div className="text-muted text-small">No scores yet</div>
              ) : (
                leaderboard.map((entry, index) => (
                  <div
                    key={entry.player_id}
                    className={`leaderboard-entry ${entry.player_id === playerId ? "leaderboard-entry--self" : ""}`}
                  >
                    <span className="leaderboard-rank">
                      {index === 0 ? "🥇" : index === 1 ? "🥈" : index === 2 ? "🥉" : `#${index + 1}`}
                    </span>
                    <span className="leaderboard-name">{entry.nickname}</span>
                    <span className="leaderboard-score">{entry.total_score}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

        <div className="waiting-screen">
          <div className="waiting-icon">🎨</div>
          <h2>Waiting for the host to start a round</h2>
          <p className="waiting-text">Get ready to create some art!</p>
        </div>
      </div>
    </div>
  );
}
