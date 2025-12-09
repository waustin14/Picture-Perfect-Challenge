import { useState } from "react";
import type { FormEvent} from "react";
import { useNavigate } from "react-router-dom";
import { apiPost } from "../api/client";
import type { GameSummary } from "../types/api";

export function AdminCreateGamePage() {
  const [adminName, setAdminName] = useState("");
  const [gameName, setGameName] = useState("");
  const [createdGame, setCreatedGame] = useState<GameSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const game = await apiPost<GameSummary>("/games", {
        admin_name: adminName,
        name: gameName,
      });
      setCreatedGame(game);
      localStorage.setItem("adminGameId", game.id);
      localStorage.setItem("adminGameCode", game.code);
    } finally {
      setLoading(false);
    }

  };

  return (
    <div className="container py-5">
      <h1 className="mb-4">Admin: Create Game</h1>
      <form onSubmit={handleSubmit} className="card p-4 mb-4">
        <div className="mb-3">
          <label className="form-label">Admin Name</label>
          <input
            className="form-control"
            value={adminName}
            onChange={(e) => setAdminName(e.target.value)}
            required
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Game Name</label>
          <input
            className="form-control"
            value={gameName}
            onChange={(e) => setGameName(e.target.value)}
            required
          />
        </div>
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? "Creating..." : "Create Game"}
        </button>
      </form>

      {createdGame && (
        <div className="card p-4">
          <h2>Game Created</h2>
          <p>Code: <strong>{createdGame.code}</strong></p>
          <button
            className="btn btn-success"
            onClick={() => navigate(`/admin/game/${createdGame.id}`)}
          >
            Go to Admin Game View
          </button>
        </div>
      )}
    </div>
  );
}
