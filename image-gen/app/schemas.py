from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None

    game_id: str
    round_id: str
    player_id: str

    width: int = Field(1024, ge=256, le=1024)
    height: int = Field(1024, ge=256, le=1024)
    num_inference_steps: Optional[int] = Field(None, ge=5, le=50)
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_id: str
    image_url: str

    game_id: str
    round_id: str
    player_id: str

    model_id: str
    seed: int
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    duration_ms: int
