from typing import Dict, Optional, List
from datetime import datetime, timedelta
import random
import string

from .models import Game, Round, Player, ImageAttempt, PlayerScore, GameStatus, RoundStatus, new_id


class GameStateStore:
    def __init__(self) -> None:
        self.games: Dict[str, Game] = {}

    def _generate_code(self, length: int = 6) -> str:
        while True:
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
            if all(g.code != code for g in self.games.values()):
                return code

    def create_game(self, admin_name: str, name: str) -> Game:
        game_id = new_id()
        code = self._generate_code()
        game = Game(
            id=game_id,
            code=code,
            name=name,
            status=GameStatus.LOBBY,
            created_at=datetime.utcnow(),
            admin_name=admin_name,
        )
        self.games[game_id] = game
        return game

    def get_game(self, game_id: str) -> Game:
        if game_id not in self.games:
            raise KeyError("Game not found")
        return self.games[game_id]

    def join_game(self, game_id: str, nickname: str) -> Player:
        game = self.get_game(game_id)
        player_id = new_id()
        player = Player(id=player_id, nickname=nickname, joined_at=datetime.utcnow())
        game.players[player_id] = player
        return player

    def create_round(self, game_id: str, reference_image_id: str, duration_seconds: int) -> Round:
        game = self.get_game(game_id)
        round_id = new_id()
        round_number = len(game.rounds) + 1
        rnd = Round(
            id=round_id,
            game_id=game_id,
            round_number=round_number,
            reference_image_id=reference_image_id,
            status=RoundStatus.PENDING,
            duration_seconds=duration_seconds,
        )
        game.rounds[round_id] = rnd
        return rnd
    
    def get_round(self, round_id: str):
        """
        Return (game, round) for a given round_id.
        Raises KeyError if not found.
        """
        for game in self.games.values():
            if round_id in game.rounds:
                return game, game.rounds[round_id]
        raise KeyError("Round not found")

    def start_round(self, round_id: str) -> Round:
        # naive search by round_id across games
        for game in self.games.values():
            if round_id in game.rounds:
                rnd = game.rounds[round_id]
                now = datetime.utcnow()
                rnd.status = RoundStatus.RUNNING
                rnd.starts_at = now
                rnd.ends_at = now + timedelta(seconds=rnd.duration_seconds)
                game.active_round_id = rnd.id
                game.status = GameStatus.ACTIVE
                return rnd
        raise KeyError("Round not found")

    def end_round(self, round_id: str) -> Round:
        for game in self.games.values():
            if round_id in game.rounds:
                rnd = game.rounds[round_id]
                rnd.status = RoundStatus.SCORING
                game.active_round_id = None
                return rnd
        raise KeyError("Round not found")

    def score_round(self, round_id: str) -> Round:
        """
        Generate fake similarity scores and compute points for each player's submission.
        Returns the round with player_scores populated.
        """
        for game in self.games.values():
            if round_id in game.rounds:
                rnd = game.rounds[round_id]
                if rnd.status != RoundStatus.SCORING:
                    raise ValueError(f"Round must be in SCORING status, got {rnd.status}")
                
                # Gather all submitted images
                submissions: Dict[str, ImageAttempt] = {}
                for attempt in rnd.image_attempts.values():
                    if attempt.is_submission:
                        submissions[attempt.player_id] = attempt
                
                # Generate fake similarity scores (0.0 - 1.0) and compute points
                player_scores: List[PlayerScore] = []
                for player_id, player in game.players.items():
                    attempt = submissions.get(player_id)
                    if attempt:
                        # Fake similarity: random between 0.3 and 0.95
                        similarity = round(random.uniform(0.3, 0.95), 3)
                        attempt.similarity_score = similarity
                        # Score = similarity * 1000, rounded
                        points = int(similarity * 1000)
                        attempt.score = points
                    else:
                        similarity = 0.0
                        points = 0
                    
                    player_scores.append(PlayerScore(
                        player_id=player_id,
                        nickname=player.nickname,
                        image_id=attempt.id if attempt else None,
                        similarity_score=similarity,
                        score=points,
                    ))
                
                # Sort by score descending
                player_scores.sort(key=lambda ps: ps.score, reverse=True)
                rnd.player_scores = player_scores
                rnd.status = RoundStatus.COMPLETE
                
                # Check if game should return to lobby or stay active
                # For now, return to lobby after each round
                game.status = GameStatus.LOBBY
                
                return rnd
        raise KeyError("Round not found")

    def complete_game(self, game_id: str) -> Game:
        """Mark the game as complete (no more rounds)."""
        game = self.get_game(game_id)
        game.status = GameStatus.COMPLETE
        return game

    def get_leaderboard(self, game_id: str) -> List[Dict]:
        """Aggregate scores across all completed rounds."""
        game = self.get_game(game_id)
        totals: Dict[str, int] = {}
        
        for rnd in game.rounds.values():
            if rnd.status == RoundStatus.COMPLETE:
                for ps in rnd.player_scores:
                    totals[ps.player_id] = totals.get(ps.player_id, 0) + ps.score
        
        leaderboard = []
        for player_id, player in game.players.items():
            leaderboard.append({
                "player_id": player_id,
                "nickname": player.nickname,
                "total_score": totals.get(player_id, 0),
            })
        
        leaderboard.sort(key=lambda x: x["total_score"], reverse=True)
        return leaderboard

    def add_image_attempt(self, round_id: str, player_id: str, prompt: str) -> ImageAttempt:
        for game in self.games.values():
            if round_id in game.rounds:
                rnd = game.rounds[round_id]
                # enforce max 6 attempts per player per round
                attempts = [a for a in rnd.image_attempts.values() if a.player_id == player_id]
                if len(attempts) >= 6:
                    raise ValueError("Max attempts reached for this round")

                image_id = new_id()
                attempt = ImageAttempt(
                    id=image_id,
                    player_id=player_id,
                    round_id=round_id,
                    prompt=prompt,
                    created_at=datetime.utcnow(),
                )
                rnd.image_attempts[image_id] = attempt
                return attempt
        raise KeyError("Round not found")

    def submit_image(self, round_id: str, player_id: str, image_id: str) -> ImageAttempt:
        for game in self.games.values():
            if round_id in game.rounds:
                rnd = game.rounds[round_id]
                if image_id not in rnd.image_attempts:
                    raise KeyError("Image not found")
                attempt = rnd.image_attempts[image_id]
                if attempt.player_id != player_id:
                    raise ValueError("Player does not own this image")
                attempt.is_submission = True
                return attempt
        raise KeyError("Round not found")
