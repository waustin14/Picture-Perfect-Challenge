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
    <div className="entry-container">
      <div className="entry-card">
        <div className="entry-header">
          <h1 className="entry-logo">Picture Perfect</h1>
          <p className="entry-tagline">Generate AI images to match the target</p>
        </div>

        <form onSubmit={handleSubmit} className="entry-form">
          <div className="form-group">
            <label className="form-label">Game Code</label>
            <input
              type="text"
              className="form-input form-input--code"
              placeholder="ABCD"
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              maxLength={6}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label">Your Nickname</label>
            <input
              type="text"
              className="form-input"
              placeholder="Enter your name"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              maxLength={20}
              required
            />
          </div>

          {error && <div className="form-error">{error}</div>}

          <button
            type="submit"
            className="btn btn--primary btn--full btn--large"
            disabled={loading || !code.trim() || !nickname.trim()}
          >
            {loading ? "Joining..." : "Join Game"}
          </button>
        </form>

        <div className="entry-footer">
          <a href="/admin" className="entry-link">Host a game instead</a>
        </div>
      </div>
    </div>
  );
}
