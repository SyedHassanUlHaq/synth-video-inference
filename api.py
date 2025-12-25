from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import tempfile
import torch
import os

from validate import (
    validate_video,
    load_raft_model,
    FusedHeadModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/mnt/data2/validation/checkpoints/fused_best.pt"
RAFT_CKPT = "/mnt/data2/validation/checkpoints/raft-sintel.pth"

# ============================
# Lifespan handler
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Loading models...")

    app.state.raft_model = load_raft_model(RAFT_CKPT, DEVICE)

    fused_model = FusedHeadModel().to(DEVICE)
    fused_model.load_checkpoint(CHECKPOINT)
    fused_model.eval()
    app.state.fused_model = fused_model

    print("✅ Models loaded")

    yield  # ---- app is running ----

    # Optional cleanup
    print("🧹 Shutting down...")
    del app.state.raft_model
    del app.state.fused_model
    torch.cuda.empty_cache()

# ============================
# App instance
# ============================

app = FastAPI(
    title="Deepfake Detection API",
    lifespan=lifespan,
)

# ============================
# API Endpoint
# ============================

@app.post("/predict")
async def predict(
    video: UploadFile = File(...),
    threshold: float = 0.5,
):
    if not video.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = validate_video(
            video_path=tmp_path,
            raft_model=app.state.raft_model,
            model=app.state.fused_model,
            device=DEVICE,
            threshold=threshold,
        )
    finally:
        os.remove(tmp_path)

    return result
