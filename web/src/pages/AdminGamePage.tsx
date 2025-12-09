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

  // Reference images state
  const [referenceImages, setReferenceImages] = useState<ReferenceImageInfo[]>([]);
  const [selectedRefImage, setSelectedRefImage] = useState<string>("");
  const [useRandomImage, setUseRandomImage] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [duration, setDuration] = useState(90);
  const [creatingRound, setCreatingRound] = useState(false);
  const [endingRound, setEndingRound] = useState(false);

  // Leaderboard state
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

  // Get player nickname by ID
  const getPlayerNickname = useCallback((playerId: string): string => {
    const player = gameState?.players.find(p => p.id === playerId);
    return player?.nickname || "Unknown";
  }, [gameState]);

  useGameWebSocket(gameId ?? "", (event: GameEvent) => {
    if (event.type === "round_started") {
      setRemainingSeconds(event.duration_seconds);
      setAttempts([]); // Clear attempts for new round
      refreshGameState();
    } else if (event.type === "round_tick") {
      setRemainingSeconds(event.remaining_seconds);
    } else if (event.type === "round_ended") {
      setRemainingSeconds(0);
      refreshGameState();
    } else if (event.type === "round_results") {
      // Round scoring complete - refresh leaderboard
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

    // Validate: must have either a selected image or use random
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
      // state + timer will update via WS
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
      // optional: call /score here later
    } catch (e) {
      console.error(e);
    } finally {
      setEndingRound(false);
    }
  };

  // Filter to show only submitted images
  const submissions = useMemo(() => {
    return attempts.filter((a) => a.isSubmission);
  }, [attempts]);

  return (
    <div className="container-fluid p-4">
      <header className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <div className="h3">Logo</div>
          {gameState && (
            <div className="text-muted">
              Game Code: <strong>{gameState.game.code}</strong>
            </div>
          )}
        </div>

        <TimerDisplay remainingSeconds={remainingSeconds} />
      </header>


      <div className="row mb-4">
        <div className="col-md-3">
          <div className="card p-3">
            <h5>Leaderboard</h5>
            {leaderboard.length === 0 ? (
              <div className="text-muted">
                {gameState?.players.length ? "No scores yet" : "No players yet"}
              </div>
            ) : (
              <table className="table table-sm mb-0">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Player</th>
                    <th className="text-end">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((entry, index) => (
                    <tr key={entry.player_id}>
                      <td>
                        {index === 0 && leaderboard.length > 1 ? "🥇" :
                         index === 1 && leaderboard.length > 2 ? "🥈" :
                         index === 2 && leaderboard.length > 3 ? "🥉" :
                         index + 1}
                      </td>
                      <td>{entry.nickname}</td>
                      <td className="text-end fw-bold">{entry.total_score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {/* Show players without scores */}
            {gameState?.players.filter(p => !leaderboard.find(l => l.player_id === p.id)).map(p => (
              <div key={p.id} className="text-muted small">{p.nickname} (waiting)</div>
            ))}
          </div>
        </div>
        <div className="col-md-9">
          <div className="card p-3 mb-3">
            <h5>Round Controls</h5>
            <form onSubmit={handleNewRound}>
              {/* Reference Image Selection */}
              <div className="mb-3">
                <label className="form-label">Reference Image</label>
                <div className="d-flex gap-2 mb-2">
                  <div className="form-check">
                    <input
                      type="checkbox"
                      className="form-check-input"
                      id="useRandom"
                      checked={useRandomImage}
                      onChange={(e) => {
                        setUseRandomImage(e.target.checked);
                        if (e.target.checked) setSelectedRefImage("");
                      }}
                    />
                    <label className="form-check-label" htmlFor="useRandom">
                      Use Random Image
                    </label>
                  </div>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline-primary ms-auto"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                  >
                    {uploading ? "Uploading..." : "Upload New"}
                  </button>
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="d-none"
                    accept="image/png,image/jpeg,image/gif,image/webp"
                    onChange={handleFileUpload}
                  />
                </div>

                {!useRandomImage && (
                  <div className="d-flex flex-wrap gap-2" style={{ maxHeight: "200px", overflowY: "auto" }}>
                    {referenceImages.length === 0 ? (
                      <div className="text-muted">No reference images uploaded yet.</div>
                    ) : (
                      referenceImages.map((img) => (
                        <button
                          key={img.url}
                          type="button"
                          className={`btn p-1 ${selectedRefImage === img.url ? "btn-primary" : "btn-outline-secondary"}`}
                          onClick={() => setSelectedRefImage(img.url)}
                          title={img.name}
                          style={{ width: "80px", height: "80px" }}
                        >
                          <img
                            src={rewriteMinioUrl(img.url) || ""}
                            alt={img.name}
                            style={{ width: "100%", height: "100%", objectFit: "cover" }}
                          />
                        </button>
                      ))
                    )}
                  </div>
                )}

                {/* Preview selected image */}
                {selectedRefImage && !useRandomImage && (
                  <div className="mt-2">
                    <small className="text-muted">Selected:</small>
                    <img
                      src={rewriteMinioUrl(selectedRefImage) || ""}
                      alt="Selected reference"
                      className="d-block mt-1 rounded"
                      style={{ maxHeight: "150px", maxWidth: "100%" }}
                    />
                  </div>
                )}
              </div>

              {/* Duration and Actions */}
              <div className="row g-3 align-items-end">
                <div className="col-md-4">
                  <label className="form-label">Duration (seconds)</label>
                  <input
                    type="number"
                    className="form-control"
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    min={10}
                    max={300}
                  />
                </div>
                <div className="col-md-8 d-flex gap-2">
                  <button
                    className="btn btn-success flex-fill"
                    type="submit"
                    disabled={creatingRound || (!useRandomImage && !selectedRefImage)}
                  >
                    {creatingRound ? "Starting..." : "New Round"}
                  </button>
                  <button
                    className="btn btn-danger flex-fill"
                    type="button"
                    onClick={handleEndRound}
                    disabled={!activeRound || endingRound}
                  >
                    {endingRound ? "Ending..." : "End Round"}
                  </button>
                </div>
              </div>
            </form>
          </div>

          <div className="card p-3">
            <h5>Submitted Images</h5>
            <div className="row g-3">
              {submissions.length === 0 && (
                <div className="col-12 text-muted">
                  {activeRound ? "Waiting for player submissions..." : "Start a round to see submissions."}
                </div>
              )}
              {submissions.map((s) => (
                <div key={s.id} className="col-6 col-md-3">
                  <div className="card text-center p-2">
                    {s.imageUrl ? (
                      <img
                        src={rewriteMinioUrl(s.imageUrl) || ""}
                        alt={`Submission by ${s.playerNickname}`}
                        className="img-fluid rounded mb-2"
                        style={{ aspectRatio: "1", objectFit: "cover" }}
                      />
                    ) : (
                      <div
                        className="bg-secondary rounded mb-2 d-flex align-items-center justify-content-center text-white"
                        style={{ aspectRatio: "1" }}
                      >
                        Loading...
                      </div>
                    )}
                    <div className="fw-bold">{s.playerNickname}</div>
                    <small className="text-muted text-truncate d-block" title={s.prompt}>
                      {s.prompt}
                    </small>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
