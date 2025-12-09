"""
Pydantic models for API request/response schemas.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ImagePair(BaseModel):
    """
    A pair of images to compare for similarity.
    
    Each image can be provided as either:
    - A base64-encoded PNG/JPEG string
    - A URL (http:// or https://) pointing to the image
    """
    reference_image: str = Field(
        ...,
        description="Reference image as base64-encoded PNG/JPEG or URL"
    )
    generated_image: str = Field(
        ...,
        description="Generated image as base64-encoded PNG/JPEG or URL"
    )
    
    # Optional identifier for tracking
    pair_id: Optional[str] = Field(
        None,
        description="Optional identifier for this pair"
    )


class SimilarityRequest(BaseModel):
    """Request model for similarity computation."""
    pairs: List[ImagePair] = Field(
        ...,
        description="List of image pairs to compare",
        min_length=1
    )
    
    @field_validator("pairs")
    @classmethod
    def validate_pairs_length(cls, v):
        from .config import MAX_PAIRS_PER_REQUEST
        if len(v) > MAX_PAIRS_PER_REQUEST:
            raise ValueError(
                f"Too many pairs: {len(v)} > {MAX_PAIRS_PER_REQUEST} (max allowed)"
            )
        return v


class SimilarityScore(BaseModel):
    """Similarity score for a single image pair."""
    distance: float = Field(
        ...,
        description="Perceptual distance (0 = identical, larger = more different)",
        ge=0.0
    )
    similarity: float = Field(
        ...,
        description="Similarity score (0-1, higher = more similar)",
        gt=0.0,
        le=1.0
    )
    pair_id: Optional[str] = Field(
        None,
        description="Optional identifier echoed from request"
    )


class SimilarityResponse(BaseModel):
    """Response model for similarity computation."""
    scores: List[SimilarityScore] = Field(
        ...,
        description="Similarity scores for each input pair"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )


class ErrorDetail(BaseModel):
    """Detailed error information."""
    message: str
    pair_index: Optional[int] = None
    pair_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[List[ErrorDetail]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    device: Optional[str] = None
    model_type: Optional[str] = None
