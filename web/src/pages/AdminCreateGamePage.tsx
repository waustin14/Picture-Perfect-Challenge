import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { apiPost } from "../api/client";
import type { GameSummary } from "../types/api";

export function AdminCreateGamePage() {
  const [adminName, setAdminName] = useState("");
  const [gameName, setGameName] = useState("");
  const [createdGame, setCreatedGame] = useState<GameSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const game = await apiPost<GameSummary>("/games", {
        admin_name: adminName,
        name: gameName,
      });
      setCreatedGame(game);
      localStorage.setItem("adminGameId", game.id);
      localStorage.setItem("adminGameCode", game.code);
    } catch (err: any) {
      setError(err.message ?? "Failed to create game");
    } finally {
      setLoading(false);
    }
  };

  if (createdGame) {
    return (
      <div className="entry-container">
        <div className="entry-card">
          <div className="entry-header">
            <h1 className="entry-logo">Picture Perfect</h1>
            <p className="entry-tagline">Game created successfully!</p>
          </div>

          <div className="game-created">
            <div className="game-created-label">Share this code with players</div>
            <div className="game-created-code">{createdGame.code}</div>
            <div className="game-created-name">{createdGame.name}</div>
          </div>

          <button
            className="btn btn--primary btn--full btn--large"
            onClick={() => navigate(`/admin/game/${createdGame.id}`)}
          >
            Start Managing Game
          </button>

          <div className="entry-footer">
            <button
              className="entry-link"
              onClick={() => setCreatedGame(null)}
              style={{ background: "none", border: "none", cursor: "pointer" }}
            >
              Create another game
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="entry-container">
      <div className="entry-card">
        <div className="entry-header">
          <h1 className="entry-logo">Picture Perfect</h1>
          <p className="entry-tagline">Create a new game session</p>
        </div>

        <form onSubmit={handleSubmit} className="entry-form">
          <div className="form-group">
            <label className="form-label">Your Name</label>
            <input
              type="text"
              className="form-input"
              placeholder="Host name"
              value={adminName}
              onChange={(e) => setAdminName(e.target.value)}
              maxLength={30}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Game Name</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g., Friday Game Night"
              value={gameName}
              onChange={(e) => setGameName(e.target.value)}
              maxLength={50}
              required
            />
          </div>

          {error && <div className="form-error">{error}</div>}

          <button
            type="submit"
            className="btn btn--primary btn--full btn--large"
            disabled={loading || !adminName.trim() || !gameName.trim()}
          >
            {loading ? "Creating..." : "Create Game"}
          </button>
        </form>

        <div className="entry-footer">
          <a href="/" className="entry-link">Join a game instead</a>
        </div>
      </div>
    </div>
  );
}
