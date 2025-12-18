FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface/hub \
    HF_DATASETS_CACHE=/data/huggingface/datasets

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app

# Create directories for caching and output
RUN mkdir -p /data/images /data/huggingface/hub /data/huggingface/datasets

# Optional: Pre-download model during build (uncomment to enable)
# This makes the image larger (~12GB) but eliminates runtime download
# ARG HF_TOKEN
# RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', token='${HF_TOKEN}')"

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
