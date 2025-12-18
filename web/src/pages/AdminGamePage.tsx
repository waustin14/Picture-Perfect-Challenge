import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import type { FormEvent, ChangeEvent } from "react";
import { useParams } from "react-router-dom";
import { apiGet, apiPost, apiUpload, rewriteMinioUrl } from "../api/client";
import type { GameState, RoundInfo, ReferenceImageInfo, ReferenceImagesListResponse, LeaderboardEntry, LeaderboardResponse } from "../types/api";
import { useGameWebSocket } from "../hooks/useGameWebSocket";
import type { GameEvent } from "../hooks/useGameWebSocket";
import { TimerDisplay } from "../components/TimerDisplay";

interface ImageAttempt {
  id: string;
  playerId: string;
  playerNickname: string;
  prompt: string;
  imageUrl: string | null;
  isSubmission: boolean;
}

export function AdminGamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [remainingSeconds, setRemainingSeconds] = useState<number | null>(null);
  const [attempts, setAttempts] = useState<ImageAttempt[]>([]);

  const [referenceImages, setReferenceImages] = useState<ReferenceImageInfo[]>([]);
  const [selectedRefImage, setSelectedRefImage] = useState<string>("");
  const [useRandomImage, setUseRandomImage] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [duration, setDuration] = useState(90);
  const [creatingRound, setCreatingRound] = useState(false);
  const [endingRound, setEndingRound] = useState(false);

  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);

  const refreshGameState = useCallback(() => {
    if (!gameId) return;
    apiGet<GameState>(`/games/${gameId}/state`).then(setGameState).catch(console.error);
  }, [gameId]);

  const refreshReferenceImages = useCallback(() => {
    apiGet<ReferenceImagesListResponse>("/reference-images")
      .then((res) => setReferenceImages(res.images))
      .catch(console.error);
  }, []);

  const refreshLeaderboard = useCallback(() => {
    if (!gameId) return;
    apiGet<LeaderboardResponse>(`/games/${gameId}/leaderboard`)
      .then((res) => setLeaderboard(res.leaderboard))
      .catch(console.error);
  }, [gameId]);

  useEffect(() => {
    refreshGameState();
    refreshReferenceImages();
    refreshLeaderboard();
  }, [refreshGameState, refreshReferenceImages, refreshLeaderboard]);

  const activeRound: RoundInfo | undefined = useMemo(() => {
    if (!gameState || !gameState.active_round_id) return undefined;
    return gameState.rounds.find((r) => r.id === gameState.active_round_id);
  }, [gameState]);

  const getPlayerNickname = useCallback((playerId: string): string => {
    const player = gameState?.players.find(p => p.id === playerId);
    return player?.nickname || "Unknown";
  }, [gameState]);

  useGameWebSocket(gameId ?? "", (event: GameEvent) => {
    if (event.type === "round_started") {
      setRemainingSeconds(event.duration_seconds);
      setAttempts([]);
      refreshGameState();
    } else if (event.type === "round_tick") {
      setRemainingSeconds(event.remaining_seconds);
    } else if (event.type === "round_ended") {
      setRemainingSeconds(0);
      refreshGameState();
    } else if (event.type === "round_results") {
      refreshLeaderboard();
      refreshGameState();
    } else if (event.type === "image_attempt_created") {
      setAttempts((prev) => [
        ...prev,
        {
          id: event.image_id,
          playerId: event.player_id,
          playerNickname: getPlayerNickname(event.player_id),
          prompt: event.prompt,
          imageUrl: event.image_url ?? null,
          isSubmission: false,
        },
      ]);
    } else if (event.type === "image_submitted") {
      setAttempts((prev) =>
        prev.map((a) =>
          a.id === event.image_id ? { ...a, isSubmission: true } : a
        )
      );
    } else if (event.type === "image_unsubmitted") {
      setAttempts((prev) =>
        prev.map((a) =>
          a.id === event.image_id ? { ...a, isSubmission: false } : a
        )
      );
    }
  });

  const handleNewRound = async (e: FormEvent) => {
    e.preventDefault();
    if (!gameId) return;

    if (!useRandomImage && !selectedRefImage) {
      alert("Please select a reference image or choose 'Random'");
      return;
    }

    setCreatingRound(true);
    try {
      const round = await apiPost<RoundInfo>(`/games/${gameId}/rounds`, {
        reference_image_id: useRandomImage ? null : selectedRefImage,
        duration_seconds: duration,
        use_random_image: useRandomImage,
      });
      await apiPost(`/rounds/${round.id}/start`);
    } catch (e) {
      console.error(e);
      alert(e instanceof Error ? e.message : "Failed to create round");
    } finally {
      setCreatingRound(false);
    }
  };

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const result = await apiUpload<{ url: string; name: string }>("/reference-images", file);
      setSelectedRefImage(result.url);
      setUseRandomImage(false);
      refreshReferenceImages();
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleEndRound = async () => {
    if (!activeRound) return;
    setEndingRound(true);
    try {
      await apiPost(`/rounds/${activeRound.id}/end`);
    } catch (e) {
      console.error(e);
    } finally {
      setEndingRound(false);
    }
  };

  const submissions = useMemo(() => {
    return attempts.filter((a) => a.isSubmission);
  }, [attempts]);

  const roundNumber = gameState?.rounds.length ?? 0;

  return (
    <div className="game-container">
      {/* Header */}
      <header className="game-header">
        <div style={{ display: "flex", alignItems: "center", gap: "var(--spacing-lg)" }}>
          <div className="game-logo">Picture Perfect</div>
          {gameState && (
            <div className="game-code">{gameState.game.code}</div>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "var(--spacing-md)" }}>
          <div className="text-muted">
            Round {activeRound ? activeRound.round_number : roundNumber + 1}
          </div>
          <TimerDisplay remainingSeconds={remainingSeconds} />
        </div>
      </header>

      <div className="game-main">
        {/* Left Sidebar - Leaderboard & Players */}
        <aside className="game-sidebar">
          <div className="leaderboard">
            <div className="leaderboard-title">
              Leaderboard ({gameState?.players.length ?? 0} players)
            </div>
            <div className="leaderboard-list">
              {leaderboard.length === 0 ? (
                gameState?.players.length ? (
                  gameState.players.map(p => (
                    <div key={p.id} className="leaderboard-entry">
                      <span className="leaderboard-rank">-</span>
                      <span className="leaderboard-name">{p.nickname}</span>
                      <span className="leaderboard-score">0</span>
                    </div>
                  ))
                ) : (
                  <div className="text-muted text-small">Waiting for players to join...</div>
                )
              ) : (
                leaderboard.map((entry, index) => (
                  <div key={entry.player_id} className="leaderboard-entry">
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

          {/* Round Controls */}
          <div className="admin-controls">
            <form onSubmit={handleNewRound}>
              <div className="admin-section">
                <div className="admin-section-title">Reference Image</div>

                <label className="checkbox-group mb-sm">
                  <input
                    type="checkbox"
                    className="checkbox-input"
                    checked={useRandomImage}
                    onChange={(e) => {
                      setUseRandomImage(e.target.checked);
                      if (e.target.checked) setSelectedRefImage("");
                    }}
                  />
                  <span className="checkbox-label">Use Random</span>
                </label>

                {!useRandomImage && (
                  <div className="ref-image-grid">
                    {referenceImages.length === 0 ? (
                      <div className="text-muted text-small">No images uploaded</div>
                    ) : (
                      referenceImages.map((img) => (
                        <div
                          key={img.url}
                          className={`ref-image-thumb ${selectedRefImage === img.url ? "ref-image-thumb--selected" : ""}`}
                          onClick={() => setSelectedRefImage(img.url)}
                        >
                          <img
                            src={rewriteMinioUrl(img.url) || ""}
                            alt={img.name}
                          />
                        </div>
                      ))
                    )}
                  </div>
                )}

                <button
                  type="button"
                  className="btn btn--outline btn--full mt-sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                >
                  {uploading ? "Uploading..." : "Upload Image"}
                </button>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  accept="image/png,image/jpeg,image/gif,image/webp"
                  onChange={handleFileUpload}
                />
              </div>

              <div className="admin-section">
                <div className="admin-section-title">Duration</div>
                <div style={{ display: "flex", alignItems: "center", gap: "var(--spacing-sm)" }}>
                  <input
                    type="range"
                    min={30}
                    max={180}
                    step={15}
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span style={{ minWidth: "50px", textAlign: "right" }}>{duration}s</span>
                </div>
              </div>

              <div className="admin-section" style={{ display: "flex", gap: "var(--spacing-sm)" }}>
                <button
                  className="btn btn--secondary"
                  type="submit"
                  disabled={creatingRound || (!useRandomImage && !selectedRefImage)}
                  style={{ flex: 1 }}
                >
                  {creatingRound ? "Starting..." : "Start Round"}
                </button>
                <button
                  className="btn btn--danger"
                  type="button"
                  onClick={handleEndRound}
                  disabled={!activeRound || endingRound}
                  style={{ flex: 1 }}
                >
                  {endingRound ? "Ending..." : "End Round"}
                </button>
              </div>
            </form>
          </div>
        </aside>

        {/* Main Content - Submissions */}
        <div className="game-content">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--spacing-sm)" }}>
            <h3 style={{ margin: 0 }}>
              Submissions ({submissions.length}/{gameState?.players.length ?? 0})
            </h3>
            {activeRound && (
              <div className="text-muted text-small">
                Total images generated: {attempts.length}
              </div>
            )}
          </div>

          {submissions.length === 0 ? (
            <div className="waiting-screen">
              {activeRound ? (
                <>
                  <div className="waiting-icon">⏳</div>
                  <p className="waiting-text">Waiting for player submissions...</p>
                </>
              ) : (
                <>
                  <div className="waiting-icon">🎮</div>
                  <h2>Ready to Start</h2>
                  <p className="waiting-text">
                    {gameState?.players.length
                      ? `${gameState.players.length} player${gameState.players.length > 1 ? 's' : ''} connected`
                      : "Waiting for players to join"}
                  </p>
                  <p className="text-muted">Share the game code with players to join</p>
                </>
              )}
            </div>
          ) : (
            <div className="submissions-grid">
              {submissions.map((s) => (
                <div key={s.id} className="submission-card">
                  {s.imageUrl ? (
                    <img
                      src={rewriteMinioUrl(s.imageUrl) || ""}
                      alt={`${s.playerNickname}'s submission`}
                      className="submission-image"
                    />
                  ) : (
                    <div className="submission-image" style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      background: "var(--bg-card-hover)",
                      color: "var(--text-muted)"
                    }}>
                      Loading...
                    </div>
                  )}
                  <div className="submission-info">
                    <div className="submission-name">{s.playerNickname}</div>
                    <div className="submission-prompt" title={s.prompt}>{s.prompt}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
