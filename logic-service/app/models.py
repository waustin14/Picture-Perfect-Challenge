from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import uuid


class GameStatus(str, Enum):
    LOBBY = "lobby"
    ACTIVE = "active"
    COMPLETE = "complete"


class RoundStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    ENDED = "ended"
    SCORING = "scoring"
    COMPLETE = "complete"


@dataclass
class Player:
    id: str
    nickname: str
    joined_at: datetime


@dataclass
class ImageAttempt:
    id: str
    player_id: str
    round_id: str
    prompt: str
    created_at: datetime
    is_submission: bool = False
    similarity_score: Optional[float] = None  # 0.0 - 1.0 similarity to reference
    score: Optional[int] = None  # computed game points


@dataclass
class PlayerScore:
    player_id: str
    nickname: str
    image_id: Optional[str]
    similarity_score: float
    score: int


@dataclass
class Round:
    id: str
    game_id: str
    round_number: int
    reference_image_id: str
    status: RoundStatus
    duration_seconds: int
    starts_at: Optional[datetime] = None
    ends_at: Optional[datetime] = None
    image_attempts: Dict[str, ImageAttempt] = field(default_factory=dict)
    player_scores: List["PlayerScore"] = field(default_factory=list)  # populated after scoring


@dataclass
class Game:
    id: str
    code: str
    name: str
    status: GameStatus
    created_at: datetime
    admin_name: str
    players: Dict[str, Player] = field(default_factory=dict)
    rounds: Dict[str, Round] = field(default_factory=dict)
    active_round_id: Optional[str] = None


def new_id() -> str:
    return str(uuid.uuid4())