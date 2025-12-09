import asyncio
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

import httpx
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy.orm import Session

from .websocket import ConnectionManager
from . import schemas
from .models import RoundStatus
from .models import GameStatus as GameStatusEnum
from .db import Base, engine, get_db
from .models_db import GameDB, PlayerDB, RoundDB, ImageAttemptDB, PlayerScoreDB
from .schemas import JoinByCodeRequest
from .storage import storage

app = FastAPI(title="Image Game Logic Service")

ws_manager = ConnectionManager()

round_timers: Dict[str, asyncio.Task] = {}
IMAGE_GEN_URL = os.getenv("IMAGE_GEN_URL", "http://image-gen:8080").rstrip("/")
IMAGE_SIM_URL = os.getenv("IMAGE_SIM_URL", "http://image-sim:8080").rstrip("/")
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", "http://localhost:9000").rstrip("/")
MINIO_INTERNAL_URL = os.getenv("MINIO_INTERNAL_URL", "http://minio:9000").rstrip("/")
_tracing_initialized = False

# Log URL configuration at startup
print(f"[config] MINIO_PUBLIC_URL={MINIO_PUBLIC_URL}")
print(f"[config] MINIO_INTERNAL_URL={MINIO_INTERNAL_URL}")


def convert_to_internal_url(url: str) -> str:
    """Convert public MinIO URLs to internal Docker network URLs."""
    if url and MINIO_PUBLIC_URL and url.startswith(MINIO_PUBLIC_URL):
        converted = url.replace(MINIO_PUBLIC_URL, MINIO_INTERNAL_URL, 1)
        print(f"[url] Converted: {url} -> {converted}")
        return converted
    print(f"[url] No conversion needed for: {url} (PUBLIC_URL={MINIO_PUBLIC_URL})")
    return url


def _normalized_otlp_endpoint() -> str:
    """
    Build an OTLP HTTP endpoint from env vars.
    Prefers OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, then OTEL_EXPORTER_OTLP_ENDPOINT,
    and ensures it ends with /v1/traces for the HTTP exporter.
    """
    base = (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "http://otel-collector:4318"
    ).rstrip("/")
    if not base.endswith("/v1/traces"):
        base = f"{base}/v1/traces"
    return base


def _wait_for_collector(endpoint: str, timeout: int = 30) -> bool:
    """Wait for the OTEL collector to be reachable."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4318

    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def init_tracing():
    global _tracing_initialized
    if _tracing_initialized:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "logic-service")
    otlp_endpoint = _normalized_otlp_endpoint()

    # Wait for collector to be available before initializing
    collector_base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    if not _wait_for_collector(collector_base):
        print(f"[otel] Collector not reachable at {collector_base}, tracing disabled")
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        HTTPXClientInstrumentor().instrument()
        SQLAlchemyInstrumentor().instrument(engine=engine)
        _tracing_initialized = True
        print(f"[otel] Tracing initialized for {service_name} -> {otlp_endpoint}")
    except Exception as exc:
        # Avoid crashing the app if collector/exporter setup fails
        print(f"[otel] Failed to initialize tracing: {exc}")


init_tracing()


def get_round_from_db(db: Session, round_id: str) -> tuple[GameDB, RoundDB]:
    """Helper to get round and game from database."""
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise KeyError("Round not found")
    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise KeyError("Game not found")
    return game, rnd


async def request_image_generation(
    game_id: str,
    round_id: str,
    player_id: str,
    prompt: str,
) -> dict:
    """
    Call the image generation service and return its JSON payload.
    """
    url = f"{IMAGE_GEN_URL}/generate"
    payload = {
        "prompt": prompt,
        "game_id": game_id,
        "round_id": round_id,
        "player_id": player_id,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise HTTPException(
            status_code=502,
            detail=f"Image generation failed ({exc.response.status_code}): {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not reach image generation service",
        ) from exc

    try:
        data = resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Image generation service returned non-JSON response",
        ) from exc

    if "image_url" not in data:
        raise HTTPException(
            status_code=502,
            detail="Image generation service returned an invalid response",
        )
    return data


async def request_image_similarity(
    pairs: list[dict],
) -> dict:
    """
    Call the image similarity service to compute similarity scores.

    Args:
        pairs: List of dicts with 'reference_image' and 'generated_image' URLs,
               and optional 'pair_id' for tracking.

    Returns:
        Response dict with 'scores' list containing similarity results.
    """
    url = f"{IMAGE_SIM_URL}/similarity"
    payload = {"pairs": pairs}

    # Debug: log the URLs being sent
    if pairs:
        print(f"[scoring] Sending {len(pairs)} pairs to similarity service")
        print(f"[scoring] Reference URL: {pairs[0].get('reference_image', 'N/A')}")
        print(f"[scoring] Generated URL: {pairs[0].get('generated_image', 'N/A')}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        print(f"[scoring] Image similarity failed ({exc.response.status_code}): {detail}")
        return None
    except httpx.RequestError as exc:
        print(f"[scoring] Could not reach image similarity service: {exc}")
        return None

    try:
        data = resp.json()
    except ValueError:
        print("[scoring] Image similarity service returned non-JSON response")
        return None

    return data


# Background task to manage round timers
async def run_round_timer(round_id: str):
    """
    Periodically broadcasts remaining time for a round, auto-ends it, and triggers scoring.
    """
    from .db import SessionLocal

    db = SessionLocal()
    try:
        rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
        if not rnd:
            return
        game_id = rnd.game_id
    finally:
        db.close()

    try:
        while True:
            db = SessionLocal()
            try:
                rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
                if not rnd:
                    break

                if rnd.status != RoundStatus.RUNNING.value or rnd.ends_at is None:
                    break

                now = datetime.utcnow()
                remaining = int((rnd.ends_at - now).total_seconds())

                if remaining <= 0:
                    # Time's up – mark round ended (transitions to SCORING)
                    rnd.status = RoundStatus.SCORING.value
                    db.commit()

                    await ws_manager.broadcast(
                        game_id,
                        {
                            "type": "round_ended",
                            "round_id": rnd.id,
                            "round_number": rnd.round_number,
                            "status": "scoring",
                        },
                    )

                    # Automatically trigger scoring
                    try:
                        print(f"[timer] Auto-scoring round {round_id}")
                        await _perform_scoring(round_id, db)
                    except Exception as e:
                        print(f"[timer] Auto-scoring failed for round {round_id}: {e}")

                    break

                # Broadcast tick
                await ws_manager.broadcast(
                    game_id,
                    {
                        "type": "round_tick",
                        "round_id": rnd.id,
                        "round_number": rnd.round_number,
                        "remaining_seconds": remaining,
                    },
                )
            finally:
                db.close()

            await asyncio.sleep(1)
    finally:
        # Clean up task reference
        round_timers.pop(round_id, None)

# CORS for your web UI during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    # Create DB tables
    Base.metadata.create_all(bind=engine)

    # Resume timers for any running rounds, or mark expired rounds as ended
    from .db import SessionLocal
    db = SessionLocal()
    try:
        running_rounds = db.query(RoundDB).filter(
            RoundDB.status == RoundStatus.RUNNING.value
        ).all()

        now = datetime.utcnow()
        for rnd in running_rounds:
            if rnd.ends_at and rnd.ends_at <= now:
                # Round has expired, mark as scoring
                rnd.status = RoundStatus.SCORING.value
                print(f"[startup] Marked expired round {rnd.id} as scoring")
            else:
                # Round still active, restart timer
                if rnd.id not in round_timers:
                    round_timers[rnd.id] = asyncio.create_task(run_round_timer(rnd.id))
                    print(f"[startup] Resumed timer for round {rnd.id}")
        db.commit()
    finally:
        db.close()


@app.post("/games", response_model=schemas.GameSummary)
def create_game(payload: schemas.CreateGameRequest, db: Session = Depends(get_db)):
    import random, string
    def generate_code():
        while True:
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            existing = db.query(GameDB).filter(GameDB.code == code).first()
            if not existing:
                return code
    
    code = generate_code()
    game = GameDB(
        code=code,
        name=payload.name,
        status=GameStatusEnum.LOBBY.value,
        admin_name=payload.admin_name,
    )
    db.add(game)
    db.commit()
    db.refresh(game)

    return schemas.GameSummary(
        id=game.id,
        code=game.code,
        name=game.name,
        status=game.status,
        created_at=game.created_at,
        admin_name=game.admin_name,
    )


@app.get("/games/by-code/{code}", response_model=schemas.GameSummary)
def get_game_by_code(code: str, db: Session = Depends(get_db)):
    game = db.query(GameDB).filter(GameDB.code == code.upper()).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return schemas.GameSummary(
        id=game.id,
        code=game.code,
        name=game.name,
        status=game.status,
        created_at=game.created_at,
        admin_name=game.admin_name,
    )


@app.post("/games/join-by-code", response_model=schemas.PlayerInfo)
def join_game_by_code(payload: JoinByCodeRequest, db: Session = Depends(get_db)):
    code = payload.code.upper()
    game = db.query(GameDB).filter(GameDB.code == code).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    player = PlayerDB(game_id=game.id, nickname=payload.nickname)
    db.add(player)
    db.commit()
    db.refresh(player)

    return schemas.PlayerInfo(
        id=player.id,
        nickname=player.nickname,
        joined_at=player.joined_at,
    )


@app.post("/games/{game_id}/join", response_model=schemas.PlayerInfo)
def join_game(game_id: str, payload: schemas.JoinGameRequest, db: Session = Depends(get_db)):
    game = db.query(GameDB).filter(GameDB.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    player = PlayerDB(game_id=game.id, nickname=payload.nickname)
    db.add(player)
    db.commit()
    db.refresh(player)
    
    return schemas.PlayerInfo(
        id=player.id,
        nickname=player.nickname,
        joined_at=player.joined_at,
    )


@app.post("/games/{game_id}/rounds", response_model=schemas.RoundInfo)
def create_round(game_id: str, payload: schemas.CreateRoundRequest, db: Session = Depends(get_db)):
    game = db.query(GameDB).filter(GameDB.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Determine reference image URL
    if payload.use_random_image:
        reference_image_url = storage.get_random_reference_image()
        if not reference_image_url:
            raise HTTPException(
                status_code=400,
                detail="No reference images available. Please upload some first.",
            )
    elif payload.reference_image_id:
        reference_image_url = payload.reference_image_id
    else:
        raise HTTPException(
            status_code=400,
            detail="Either reference_image_id or use_random_image must be provided",
        )

    # Get next round number
    round_count = db.query(RoundDB).filter(RoundDB.game_id == game_id).count()
    round_number = round_count + 1

    rnd = RoundDB(
        game_id=game_id,
        round_number=round_number,
        reference_image_id=reference_image_url,
        status=RoundStatus.PENDING.value,
        duration_seconds=payload.duration_seconds,
    )
    db.add(rnd)
    db.commit()
    db.refresh(rnd)

    return schemas.RoundInfo(
        id=rnd.id,
        game_id=rnd.game_id,
        round_number=rnd.round_number,
        reference_image_id=rnd.reference_image_id,
        status=RoundStatus(rnd.status),
        duration_seconds=rnd.duration_seconds,
        starts_at=rnd.starts_at,
        ends_at=rnd.ends_at,
    )


@app.post("/rounds/{round_id}/start", response_model=schemas.RoundInfo)
async def start_round(round_id: str, db: Session = Depends(get_db)):
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")
    
    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    now = datetime.utcnow()
    rnd.status = RoundStatus.RUNNING.value
    rnd.starts_at = now
    rnd.ends_at = now + timedelta(seconds=rnd.duration_seconds)
    game.status = GameStatusEnum.ACTIVE.value
    db.commit()
    db.refresh(rnd)

    # broadcast event
    await ws_manager.broadcast(
        game.id,
        {
            "type": "round_started",
            "round_id": rnd.id,
            "round_number": rnd.round_number,
            "duration_seconds": rnd.duration_seconds,
            "starts_at": rnd.starts_at.isoformat() if rnd.starts_at else None,
            "ends_at": rnd.ends_at.isoformat() if rnd.ends_at else None,
            "reference_image_url": rnd.reference_image_id,
        },
    )

    # start timer task (if one isn't already running for this round)
    if round_id not in round_timers:
        round_timers[round_id] = asyncio.create_task(run_round_timer(round_id))

    return schemas.RoundInfo(
        id=rnd.id,
        game_id=rnd.game_id,
        round_number=rnd.round_number,
        reference_image_id=rnd.reference_image_id,
        status=RoundStatus(rnd.status),
        duration_seconds=rnd.duration_seconds,
        starts_at=rnd.starts_at,
        ends_at=rnd.ends_at,
    )


@app.post("/rounds/{round_id}/end", response_model=schemas.RoundResultsInfo)
async def end_round_manual(round_id: str, db: Session = Depends(get_db)):
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")

    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Optional: only allow ending if it's currently running or pending
    if rnd.status not in {RoundStatus.RUNNING.value, RoundStatus.PENDING.value}:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot end round in status {rnd.status}",
        )

    # Mark as ended in the database
    rnd.status = RoundStatus.SCORING.value
    db.commit()

    # Cancel timer task if it's running
    task = round_timers.pop(round_id, None)
    if task is not None:
        task.cancel()

    # Broadcast to all clients in the game
    await ws_manager.broadcast(
        game.id,
        {
            "type": "round_ended",
            "round_id": rnd.id,
            "round_number": rnd.round_number,
            "ended_by": "admin",
            "status": "scoring",
        },
    )

    # Automatically trigger scoring
    print(f"[admin] Auto-scoring round {round_id}")
    return await _perform_scoring(round_id, db)


@app.post("/rounds/{round_id}/generate", response_model=schemas.GenerateImageResponse)
async def generate_image(round_id: str, payload: schemas.GenerateImageRequest, db: Session = Depends(get_db)):
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")
    
    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    player = db.query(PlayerDB).filter(
        PlayerDB.id == payload.player_id,
        PlayerDB.game_id == game.id,
    ).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found in this game")

    if rnd.status != RoundStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="Round is not running")

    if rnd.ends_at and datetime.utcnow() > rnd.ends_at:
        raise HTTPException(status_code=400, detail="Round has ended")
    
    # Check max attempts (6 per player per round)
    attempt_count = db.query(ImageAttemptDB).filter(
        ImageAttemptDB.round_id == round_id,
        ImageAttemptDB.player_id == payload.player_id
    ).count()
    if attempt_count >= 6:
        raise HTTPException(status_code=400, detail="Max attempts reached for this round")
    
    attempt = ImageAttemptDB(
        round_id=round_id,
        player_id=payload.player_id,
        prompt=payload.prompt,
    )
    db.add(attempt)
    db.flush()

    try:
        gen_response = await request_image_generation(
            game_id=game.id,
            round_id=rnd.id,
            player_id=player.id,
            prompt=payload.prompt,
        )
    except Exception:
        db.rollback()
        raise

    attempt.image_url = gen_response.get("image_url")
    db.commit()
    db.refresh(attempt)

    await ws_manager.broadcast(
        game.id,
        {
            "type": "image_attempt_created",
            "round_id": attempt.round_id,
            "player_id": attempt.player_id,
            "image_id": attempt.id,
            "prompt": attempt.prompt,
            "created_at": attempt.created_at.isoformat(),
            "image_url": attempt.image_url,
        },
    )

    return schemas.GenerateImageResponse(
        image_id=attempt.id,
        player_id=attempt.player_id,
        round_id=attempt.round_id,
        prompt=attempt.prompt,
        created_at=attempt.created_at,
        image_url=attempt.image_url,
    )


@app.post("/rounds/{round_id}/submit")
async def submit_image(round_id: str, payload: schemas.SubmitImageRequest, db: Session = Depends(get_db)):
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")

    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    attempt = db.query(ImageAttemptDB).filter(ImageAttemptDB.id == payload.image_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Image not found")

    if attempt.player_id != payload.player_id:
        raise HTTPException(status_code=400, detail="Player does not own this image")

    if attempt.round_id != round_id:
        raise HTTPException(status_code=400, detail="Image is not from this round")

    # Find any previous submissions by this player for this round
    previous_submissions = db.query(ImageAttemptDB).filter(
        ImageAttemptDB.round_id == round_id,
        ImageAttemptDB.player_id == payload.player_id,
        ImageAttemptDB.is_submission == True,
        ImageAttemptDB.id != payload.image_id,
    ).all()

    # Un-mark previous submissions
    for prev in previous_submissions:
        prev.is_submission = False

    # Mark the new submission
    attempt.is_submission = True
    db.commit()

    # Broadcast un-submission events for previous submissions
    for prev in previous_submissions:
        await ws_manager.broadcast(
            game.id,
            {
                "type": "image_unsubmitted",
                "round_id": prev.round_id,
                "player_id": prev.player_id,
                "image_id": prev.id,
            },
        )

    # Broadcast the new submission
    await ws_manager.broadcast(
        game.id,
        {
            "type": "image_submitted",
            "round_id": attempt.round_id,
            "player_id": attempt.player_id,
            "image_id": attempt.id,
        },
    )
    return {"status": "ok"}


async def _perform_scoring(round_id: str, db: Session) -> schemas.RoundResultsInfo:
    """
    Internal function to perform scoring for a round.
    Calls image-similarity service for real scores, falls back to random if unavailable.
    """
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")

    game = db.query(GameDB).filter(GameDB.id == rnd.game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Get all players in the game
    players = db.query(PlayerDB).filter(PlayerDB.game_id == game.id).all()

    # Get all submitted images for this round
    submissions = db.query(ImageAttemptDB).filter(
        ImageAttemptDB.round_id == round_id,
        ImageAttemptDB.is_submission == True
    ).all()
    submission_by_player = {s.player_id: s for s in submissions}

    # Clear existing player scores for this round
    db.query(PlayerScoreDB).filter(PlayerScoreDB.round_id == round_id).delete()

    # Build pairs for similarity service
    # Convert URLs to internal Docker network URLs for service-to-service calls
    reference_image_url = rnd.reference_image_id  # This should be a URL
    reference_image_internal = convert_to_internal_url(reference_image_url)
    pairs = []
    submission_list = []  # Keep track of submission order

    for submission in submissions:
        if submission.image_url:
            pairs.append({
                "reference_image": reference_image_internal,
                "generated_image": convert_to_internal_url(submission.image_url),
                "pair_id": submission.id,
            })
            submission_list.append(submission)

    # Call image similarity service
    similarity_scores = {}
    if pairs and reference_image_internal and reference_image_internal.startswith("http"):
        sim_response = await request_image_similarity(pairs)
        print(f"[scoring] Raw response: {sim_response}")
        if sim_response and "scores" in sim_response:
            for score_data in sim_response["scores"]:
                pair_id = score_data.get("pair_id")
                similarity = score_data.get("similarity", 0.0)
                print(f"[scoring] Score data: pair_id={pair_id}, similarity={similarity}")
                if pair_id:
                    similarity_scores[pair_id] = similarity
            print(f"[scoring] Got {len(similarity_scores)} similarity scores from service")
        else:
            print(f"[scoring] Falling back to random scores (response: {sim_response})")
    else:
        print(f"[scoring] Skipping similarity service: pairs={len(pairs)}, ref_url={reference_image_internal}")

    player_scores = []
    print(f"[scoring] Processing {len(players)} players, similarity_scores keys: {list(similarity_scores.keys())}")
    for player in players:
        attempt = submission_by_player.get(player.id)
        if attempt:
            print(f"[scoring] Player {player.nickname}: attempt.id={attempt.id}, in similarity_scores={attempt.id in similarity_scores}")
            # Use real similarity if available, otherwise random fallback
            if attempt.id in similarity_scores:
                similarity = round(similarity_scores[attempt.id], 3)
                print(f"[scoring] Using real similarity: {similarity}")
            else:
                similarity = round(random.uniform(0.3, 0.95), 3)
                print(f"[scoring] Using random similarity: {similarity}")
            points = int(similarity * 1000)
            attempt.similarity_score = similarity
            attempt.score = points
        else:
            print(f"[scoring] Player {player.nickname}: no submission")
            similarity = 0.0
            points = 0

        # Create PlayerScoreDB entry
        player_score = PlayerScoreDB(
            round_id=round_id,
            player_id=player.id,
            image_attempt_id=attempt.id if attempt else None,
            similarity_score=similarity,
            score=points,
        )
        db.add(player_score)

        player_scores.append(schemas.PlayerScoreInfo(
            player_id=player.id,
            nickname=player.nickname,
            image_id=attempt.id if attempt else None,
            similarity_score=similarity,
            score=points,
        ))

    # Sort by score descending
    player_scores.sort(key=lambda ps: ps.score, reverse=True)

    # Build image_id -> image_url mapping for broadcast
    image_url_by_id = {attempt.id: attempt.image_url for attempt in submission_by_player.values()}

    rnd.status = RoundStatus.COMPLETE.value
    game.status = GameStatusEnum.LOBBY.value
    db.commit()

    # Broadcast results to all clients
    await ws_manager.broadcast(
        game.id,
        {
            "type": "round_results",
            "round_id": rnd.id,
            "round_number": rnd.round_number,
            "reference_image_url": rnd.reference_image_id,
            "status": rnd.status,
            "player_scores": [
                {
                    "player_id": ps.player_id,
                    "nickname": ps.nickname,
                    "image_id": ps.image_id,
                    "image_url": image_url_by_id.get(ps.image_id) if ps.image_id else None,
                    "similarity_score": ps.similarity_score,
                    "score": ps.score,
                }
                for ps in player_scores
            ],
            "game_status": game.status,
        },
    )

    return schemas.RoundResultsInfo(
        round_id=rnd.id,
        round_number=rnd.round_number,
        status=RoundStatus(rnd.status),
        player_scores=player_scores,
    )


@app.post("/rounds/{round_id}/score", response_model=schemas.RoundResultsInfo)
async def score_round(round_id: str, db: Session = Depends(get_db)):
    """
    Trigger scoring for a round that has ended.
    Uses image-similarity service for real scores, falls back to random if unavailable.
    """
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")

    if rnd.status != RoundStatus.SCORING.value:
        raise HTTPException(status_code=400, detail=f"Round must be in SCORING status, got {rnd.status}")

    return await _perform_scoring(round_id, db)


@app.get("/rounds/{round_id}/results", response_model=schemas.RoundResultsInfo)
def get_round_results(round_id: str, db: Session = Depends(get_db)):
    """Get the results of a completed round."""
    rnd = db.query(RoundDB).filter(RoundDB.id == round_id).first()
    if not rnd:
        raise HTTPException(status_code=404, detail="Round not found")

    if rnd.status != RoundStatus.COMPLETE.value:
        raise HTTPException(status_code=400, detail="Round not yet scored")

    # Get player scores from database
    scores = db.query(PlayerScoreDB).filter(PlayerScoreDB.round_id == round_id).all()
    
    player_scores = []
    for score in scores:
        player = db.query(PlayerDB).filter(PlayerDB.id == score.player_id).first()
        player_scores.append(schemas.PlayerScoreInfo(
            player_id=score.player_id,
            nickname=player.nickname if player else "Unknown",
            image_id=score.image_attempt_id,
            similarity_score=score.similarity_score,
            score=score.score,
        ))
    
    # Sort by score descending
    player_scores.sort(key=lambda ps: ps.score, reverse=True)

    return schemas.RoundResultsInfo(
        round_id=rnd.id,
        round_number=rnd.round_number,
        status=RoundStatus(rnd.status),
        player_scores=player_scores,
    )


@app.get("/games/{game_id}/leaderboard", response_model=schemas.LeaderboardResponse)
def get_leaderboard(game_id: str, db: Session = Depends(get_db)):
    """Get the aggregated leaderboard across all completed rounds."""
    game = db.query(GameDB).filter(GameDB.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    # Get all completed rounds for this game
    completed_rounds = db.query(RoundDB).filter(
        RoundDB.game_id == game_id,
        RoundDB.status == RoundStatus.COMPLETE.value
    ).all()
    
    # Aggregate scores by player
    totals: Dict[str, int] = {}
    for rnd in completed_rounds:
        scores = db.query(PlayerScoreDB).filter(PlayerScoreDB.round_id == rnd.id).all()
        for score in scores:
            totals[score.player_id] = totals.get(score.player_id, 0) + score.score
    
    # Build leaderboard with player info
    players = db.query(PlayerDB).filter(PlayerDB.game_id == game_id).all()
    leaderboard = []
    for player in players:
        leaderboard.append({
            "player_id": player.id,
            "nickname": player.nickname,
            "total_score": totals.get(player.id, 0),
        })
    
    leaderboard.sort(key=lambda x: x["total_score"], reverse=True)

    return schemas.LeaderboardResponse(
        game_id=game_id,
        leaderboard=[
            schemas.LeaderboardEntry(
                player_id=entry["player_id"],
                nickname=entry["nickname"],
                total_score=entry["total_score"],
            )
            for entry in leaderboard
        ],
    )


@app.post("/games/{game_id}/complete")
async def complete_game(game_id: str, db: Session = Depends(get_db)):
    """Mark the game as complete (no more rounds)."""
    game = db.query(GameDB).filter(GameDB.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game.status = GameStatusEnum.COMPLETE.value
    db.commit()
    
    # Get leaderboard for broadcast
    completed_rounds = db.query(RoundDB).filter(
        RoundDB.game_id == game_id,
        RoundDB.status == RoundStatus.COMPLETE.value
    ).all()
    
    totals: Dict[str, int] = {}
    for rnd in completed_rounds:
        scores = db.query(PlayerScoreDB).filter(PlayerScoreDB.round_id == rnd.id).all()
        for score in scores:
            totals[score.player_id] = totals.get(score.player_id, 0) + score.score
    
    players = db.query(PlayerDB).filter(PlayerDB.game_id == game_id).all()
    leaderboard = []
    for player in players:
        leaderboard.append({
            "player_id": player.id,
            "nickname": player.nickname,
            "total_score": totals.get(player.id, 0),
        })
    leaderboard.sort(key=lambda x: x["total_score"], reverse=True)

    await ws_manager.broadcast(
        game_id,
        {
            "type": "game_complete",
            "game_id": game.id,
            "leaderboard": leaderboard,
        },
    )

    return {"status": "ok", "game_status": game.status}


@app.get("/games/{game_id}/state", response_model=schemas.GameState)
def get_game_state(game_id: str, db: Session = Depends(get_db)):
    game = db.query(GameDB).filter(GameDB.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    # Determine active_round_id (first round with status RUNNING and not expired)
    active_round_id = None
    now = datetime.utcnow()
    for r in game.rounds:
        if r.status == RoundStatus.RUNNING.value:
            # Also check if the round hasn't expired by time
            if r.ends_at is None or r.ends_at > now:
                active_round_id = r.id
                break

    game_summary = schemas.GameSummary(
        id=game.id,
        code=game.code,
        name=game.name,
        status=game.status,
        created_at=game.created_at,
        admin_name=game.admin_name,
    )

    players = [
        schemas.PlayerInfo(
            id=p.id,
            nickname=p.nickname,
            joined_at=p.joined_at,
        )
        for p in game.players
    ]

    rounds = [
        schemas.RoundInfo(
            id=r.id,
            game_id=r.game_id,
            round_number=r.round_number,
            reference_image_id=r.reference_image_id,
            status=RoundStatus(r.status),
            duration_seconds=r.duration_seconds,
            starts_at=r.starts_at,
            ends_at=r.ends_at,
        )
        for r in game.rounds
    ]

    return schemas.GameState(
        game=game_summary,
        players=players,
        rounds=rounds,
        active_round_id=active_round_id,
    )


# Reference image management
@app.post("/reference-images", response_model=schemas.ReferenceImageUploadResponse)
async def upload_reference_image(file: UploadFile = File(...)):
    """
    Upload a reference image to be used in rounds.
    Accepts PNG, JPEG, GIF, and WebP images.
    """
    # Validate content type
    allowed_types = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}",
        )

    # Read file content
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    # Upload to MinIO
    url = storage.upload_reference_image(
        image_bytes=content,
        filename=file.filename or "image.png",
        content_type=file.content_type,
    )

    if not url:
        raise HTTPException(status_code=500, detail="Failed to upload image")

    # Extract name from URL
    name = url.split("/")[-1]

    return schemas.ReferenceImageUploadResponse(url=url, name=name)


@app.get("/reference-images", response_model=schemas.ReferenceImagesListResponse)
def list_reference_images():
    """
    List all available reference images.
    """
    images = storage.list_reference_images()
    return schemas.ReferenceImagesListResponse(
        images=[
            schemas.ReferenceImageInfo(
                name=img["name"],
                url=img["url"],
                size=img.get("size"),
                last_modified=img.get("last_modified"),
            )
            for img in images
        ]
    )


# Health checks
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    # later you can check DB, Redis, etc.
    return {"status": "ready"}


@app.websocket("/ws/games/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    await ws_manager.connect(game_id, websocket)
    try:
        while True:
            # For now, we just keep the connection open; clients don't need to send messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(game_id, websocket)
