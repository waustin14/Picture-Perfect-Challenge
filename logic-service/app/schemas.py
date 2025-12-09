from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from .models import GameStatus, RoundStatus


class CreateGameRequest(BaseModel):
    admin_name: str
    name: str


class GameSummary(BaseModel):
    id: str
    code: str
    name: str
    status: GameStatus
    created_at: datetime
    admin_name: str


class PlayerInfo(BaseModel):
    id: str
    nickname: str
    joined_at: datetime


class JoinGameRequest(BaseModel):
    nickname: str


class JoinByCodeRequest(BaseModel):
    code: str
    nickname: str


class CreateRoundRequest(BaseModel):
    reference_image_id: Optional[str] = None  # URL or None for random
    duration_seconds: int
    use_random_image: bool = False  # If True, picks a random reference image


class RoundInfo(BaseModel):
    id: str
    game_id: str
    round_number: int
    reference_image_id: str
    status: RoundStatus
    duration_seconds: int
    starts_at: Optional[datetime]
    ends_at: Optional[datetime]


class PlayerScoreInfo(BaseModel):
    player_id: str
    nickname: str
    image_id: Optional[str]
    similarity_score: float
    score: int


class RoundResultsInfo(BaseModel):
    round_id: str
    round_number: int
    status: RoundStatus
    player_scores: List[PlayerScoreInfo]


class LeaderboardEntry(BaseModel):
    player_id: str
    nickname: str
    total_score: int


class LeaderboardResponse(BaseModel):
    game_id: str
    leaderboard: List[LeaderboardEntry]


class GenerateImageRequest(BaseModel):
    player_id: str
    prompt: str


class GenerateImageResponse(BaseModel):
    image_id: str
    player_id: str
    round_id: str
    prompt: str
    created_at: datetime
    image_url: Optional[str]


class SubmitImageRequest(BaseModel):
    player_id: str
    image_id: str


class GameState(BaseModel):
    game: GameSummary
    players: List[PlayerInfo]
    rounds: List[RoundInfo]
    active_round_id: Optional[str]


class ReferenceImageInfo(BaseModel):
    name: str
    url: str
    size: Optional[int] = None
    last_modified: Optional[str] = None


class ReferenceImageUploadResponse(BaseModel):
    url: str
    name: str


class ReferenceImagesListResponse(BaseModel):
    images: List[ReferenceImageInfo]
