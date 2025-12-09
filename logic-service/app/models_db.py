from datetime import datetime
from typing import List, Optional
import uuid
from zoneinfo import ZoneInfo

from sqlalchemy import String, Text, Integer, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class GameDB(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    code: Mapped[str] = mapped_column(String(8), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    admin_name: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.now(ZoneInfo("UTC")))

    players: Mapped[List["PlayerDB"]] = relationship(back_populates="game", cascade="all, delete-orphan")
    rounds: Mapped[List["RoundDB"]] = relationship(back_populates="game", cascade="all, delete-orphan")


class PlayerDB(Base):
    __tablename__ = "players"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.id", ondelete="CASCADE"))
    nickname: Mapped[str] = mapped_column(Text, nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    game: Mapped[GameDB] = relationship(back_populates="players")


class RoundDB(Base):
    __tablename__ = "rounds"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.id", ondelete="CASCADE"))
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    reference_image_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    starts_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    game: Mapped[GameDB] = relationship(back_populates="rounds")
    image_attempts: Mapped[List["ImageAttemptDB"]] = relationship(
        back_populates="round", cascade="all, delete-orphan"
    )


class ImageAttemptDB(Base):
    __tablename__ = "image_attempts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    round_id: Mapped[str] = mapped_column(String, ForeignKey("rounds.id", ondelete="CASCADE"))
    player_id: Mapped[str] = mapped_column(String, ForeignKey("players.id", ondelete="CASCADE"))
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    is_submission: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    similarity_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    round: Mapped[RoundDB] = relationship(back_populates="image_attempts")
    player: Mapped[PlayerDB] = relationship()


class PlayerScoreDB(Base):
    __tablename__ = "player_scores"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    round_id: Mapped[str] = mapped_column(String, ForeignKey("rounds.id", ondelete="CASCADE"))
    player_id: Mapped[str] = mapped_column(String, ForeignKey("players.id", ondelete="CASCADE"))
    image_attempt_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("image_attempts.id", ondelete="SET NULL"), nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    round: Mapped[RoundDB] = relationship()
    player: Mapped[PlayerDB] = relationship()

