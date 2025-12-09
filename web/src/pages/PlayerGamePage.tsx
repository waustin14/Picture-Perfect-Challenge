import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { apiGet, apiPost } from "../api/client";
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

  // Handle WebSocket game events
  useGameWebSocket(gameId ?? "", (event: GameEvent) => {
    if (event.type === "round_started") {
      setRemainingSeconds(event.duration_seconds);
      setAttempts([]);
      setSelectedImageId(null);
      setReferenceImageUrl(event.reference_image_url);
      setRoundResults(null); // Clear previous round results

      // Pull fresh state so active_round_id and rounds are up to date
      refreshGameState();
    } else if (event.type === "round_tick") {
      setRemainingSeconds(event.remaining_seconds);
    } else if (event.type === "round_ended") {
      setRemainingSeconds(0);
      // optional: refresh state again if you want latest statuses
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
        // Clear selection since the image is now submitted
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
      // Show round results screen
      setRoundResults({
        round_number: event.round_number,
        reference_image_url: event.reference_image_url,
        player_scores: event.player_scores,
      });
      // Also refresh leaderboard
      refreshLeaderboard();
      refreshGameState();
    }
  });


  const handleGenerate = async () => {
    if (!activeRound || activeRound.status !== "running" || !prompt.trim() || !playerId) return;
    setGenerating(true);
    try {
      await apiPost("/rounds/" + activeRound.id + "/generate", {
        player_id: playerId,
        prompt,
      });
      setPrompt("");
      // The attempt will show up via WebSocket event
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

  // Get current player's score from leaderboard
  const playerScore = useMemo(() => {
    const entry = leaderboard.find((e) => e.player_id === playerId);
    return entry?.total_score ?? 0;
  }, [leaderboard, playerId]);

  return (
    <div className="container-fluid p-4">
      <header className="d-flex justify-content-between align-items-center mb-4">
        <div className="h3">Logo</div>
        <div className="d-flex align-items-center gap-3">
          <div className="text-end">
            <div>Player: {nickname}</div>
            <div className="small text-muted">Score: <strong>{playerScore}</strong></div>
          </div>
          <TimerDisplay remainingSeconds={remainingSeconds} />
        </div>
      </header>

      {/* Leaderboard */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card p-3">
            <h5 className="mb-2">Leaderboard</h5>
            {leaderboard.length === 0 ? (
              <div className="text-muted">No scores yet</div>
            ) : (
              <div className="d-flex flex-wrap gap-3">
                {leaderboard.map((entry, index) => (
                  <div
                    key={entry.player_id}
                    className={`d-flex align-items-center gap-2 px-3 py-2 rounded ${
                      entry.player_id === playerId ? "bg-primary bg-opacity-25" : "bg-light"
                    }`}
                  >
                    <span>
                      {index === 0 && leaderboard.length > 1 ? "🥇" :
                       index === 1 && leaderboard.length > 2 ? "🥈" :
                       index === 2 && leaderboard.length > 3 ? "🥉" :
                       `#${index + 1}`}
                    </span>
                    <span className={entry.player_id === playerId ? "fw-bold" : ""}>
                      {entry.nickname}
                    </span>
                    <span className="fw-bold">{entry.total_score}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Round Results Screen */}
      {roundResults ? (
        <div className="card p-4">
          <h3 className="text-center mb-4">Round {roundResults.round_number} Results</h3>

          {/* Reference Image */}
          <div className="text-center mb-4">
            <h5>Reference Image</h5>
            <img
              src={roundResults.reference_image_url}
              alt="Reference"
              className="img-fluid rounded"
              style={{ maxHeight: "200px", objectFit: "contain" }}
            />
          </div>

          {/* Top 3 Submissions */}
          <h5 className="text-center mb-3">Top Submissions</h5>
          <div className="row justify-content-center mb-4">
            {roundResults.player_scores.slice(0, 3).map((ps, index) => (
              <div key={ps.player_id} className="col-md-4 col-sm-6 mb-3">
                <div className={`card text-center p-3 ${index === 0 ? "border-warning border-2" : ""}`}>
                  <div className="mb-2">
                    <span style={{ fontSize: "2rem" }}>
                      {index === 0 ? "🥇" : index === 1 ? "🥈" : "🥉"}
                    </span>
                  </div>
                  {ps.image_url ? (
                    <img
                      src={ps.image_url}
                      alt={`${ps.nickname}'s submission`}
                      className="img-fluid rounded mb-2"
                      style={{ aspectRatio: "1", objectFit: "cover", maxHeight: "150px" }}
                    />
                  ) : (
                    <div
                      className="bg-secondary rounded mb-2 d-flex align-items-center justify-content-center text-white"
                      style={{ aspectRatio: "1", maxHeight: "150px" }}
                    >
                      No submission
                    </div>
                  )}
                  <div className="fw-bold">{ps.nickname}</div>
                  <div className="text-primary fs-4">{ps.score} pts</div>
                  <div className="text-muted small">
                    {Math.round(ps.similarity_score * 100)}% match
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Player's score if not in top 3 */}
          {(() => {
            const playerIndex = roundResults.player_scores.findIndex(ps => ps.player_id === playerId);
            const playerResult = roundResults.player_scores[playerIndex];
            if (playerIndex >= 3 && playerResult) {
              return (
                <div className="text-center border-top pt-4">
                  <h5>Your Result</h5>
                  <div className="d-inline-block">
                    <div className="card p-3">
                      <div className="text-muted mb-1">#{playerIndex + 1}</div>
                      {playerResult.image_url && (
                        <img
                          src={playerResult.image_url}
                          alt="Your submission"
                          className="img-fluid rounded mb-2"
                          style={{ maxHeight: "100px", objectFit: "cover" }}
                        />
                      )}
                      <div className="text-primary fs-4">{playerResult.score} pts</div>
                      <div className="text-muted small">
                        {Math.round(playerResult.similarity_score * 100)}% match
                      </div>
                    </div>
                  </div>
                </div>
              );
            }
            return null;
          })()}

          <div className="text-center mt-4 text-muted">
            Waiting for the next round...
          </div>
        </div>
      ) : activeRound ? (
        <>
          {/* Reference Image Display */}
          <div className="card p-3 mb-4 text-center">
            <h5>Reference Image - Try to match this!</h5>
            {referenceImageUrl ? (
              <img
                src={referenceImageUrl}
                alt="Reference"
                className="img-fluid rounded mx-auto"
                style={{ maxHeight: "250px", objectFit: "contain" }}
              />
            ) : (
              <div className="text-muted">Loading reference image...</div>
            )}
          </div>

          <div className="row mb-4">
            <div className="col-md-8">
              <div className="card p-3 text-center">
                {/* Show selected image or placeholder */}
                {selectedImageId ? (
                  <div>
                    {attempts.find(a => a.id === selectedImageId)?.imageUrl ? (
                      <img
                        src={attempts.find(a => a.id === selectedImageId)?.imageUrl || ""}
                        alt="Selected image"
                        className="img-fluid rounded"
                        style={{ maxHeight: "400px", objectFit: "contain" }}
                      />
                    ) : (
                      <div className="display-6 text-muted">Loading...</div>
                    )}
                    <div className="mt-2 text-muted small">
                      Prompt: {attempts.find(a => a.id === selectedImageId)?.prompt}
                    </div>
                  </div>
                ) : (
                  <div className="display-6 text-muted">Select an image to preview</div>
                )}
              </div>
            </div>
            <div className="col-md-4">
              <div className="d-flex flex-wrap gap-2 justify-content-end">
                {attempts.map((a, index) => (
                  <button
                    key={a.id}
                    type="button"
                    className={`btn p-1 position-relative ${
                      a.id === selectedImageId ? "btn-primary" : "btn-outline-primary"
                    } ${a.isSubmission ? "border-success border-2" : ""}`}
                    onClick={() => setSelectedImageId(a.id)}
                    disabled={a.isSubmission}
                    style={{ width: "80px", height: "80px", overflow: "hidden" }}
                    title={a.prompt}
                  >
                    {a.imageUrl ? (
                      <img
                        src={a.imageUrl}
                        alt={`Attempt ${index + 1}`}
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                    ) : (
                      <span className="small">#{index + 1}</span>
                    )}
                    {a.isSubmission && (
                      <div className="position-absolute bottom-0 start-0 end-0 bg-success text-white" style={{ fontSize: "0.6rem" }}>
                        Submitted
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="card p-3 mb-3">
            <label className="form-label">Prompt</label>
            <textarea
              className="form-control mb-3"
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={!roundRunning || roundEnded || maxAttemptsReached}
            />
            <div className="d-flex justify-content-between">
              <button
                className="btn btn-success"
                type="button"
                onClick={handleSubmitImage}
                disabled={!selectedImageId || roundEnded || submitting}
              >
                {submitting ? "Submitting..." : "Submit"}
              </button>
              <button
                className="btn btn-success"
                type="button"
                onClick={handleGenerate}
                disabled={!roundRunning || roundEnded || generating || maxAttemptsReached}
              >
                {generating ? "Generating..." : "Generate"}
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="alert alert-info">Waiting for the host to start a round…</div>
      )}
    </div>
  );
}
