import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../api/client";
import type { PlayerInfo } from "../types/api";

interface JoinResponse extends PlayerInfo {}

export function JoinPage() {
  const [code, setCode] = useState("");
  const [nickname, setNickname] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const player: JoinResponse = await apiPost("/games/join-by-code", {
        code,
        nickname,
      });

      // You need the gameId too, so hit by-code endpoint
      const gameSummary = await apiGet<{ id: string; code: string; name: string; status: string; created_at: string; admin_name: string }>(
        "/games/by-code/" + code.toUpperCase()
      );

      localStorage.setItem("playerId", player.id);
      localStorage.setItem("playerNickname", player.nickname);
      localStorage.setItem("gameId", gameSummary.id);
      localStorage.setItem("gameCode", gameSummary.code);

      navigate(`/game/${gameSummary.id}`);
    } catch (err: any) {
      setError(err.message ?? "Failed to join game");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-5">
      <h1 className="mb-4">Join Game</h1>
      <form onSubmit={handleSubmit} className="card p-4">
        <div className="mb-3">
          <label className="form-label">Game Code</label>
          <input
            className="form-control"
            value={code}
            onChange={(e) => setCode(e.target.value.toUpperCase())}
            required
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Nickname</label>
          <input
            className="form-control"
            value={nickname}
            onChange={(e) => setNickname(e.target.value)}
            required
          />
        </div>
        {error && <div className="alert alert-danger">{error}</div>}
        <button className="btn btn-success" type="submit" disabled={loading}>
          {loading ? "Joining..." : "Join"}
        </button>
      </form>
    </div>
  );
}
