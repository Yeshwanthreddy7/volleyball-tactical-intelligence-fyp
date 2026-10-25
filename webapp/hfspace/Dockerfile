# Hugging Face Space (Docker SDK): the full Volleyball Tactical Analysis app on a
# free CPU Space. Built from a staging folder made by build_hf_space.py.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2 \
    VTA_REPO_ROOT=/home/user/app VTA_STORAGE_DIR=/tmp/vta VTA_DEMOS_DIR=/home/user/app/webapp/storage/demos \
    VTA_FRONTEND_DIST=/home/user/app/webapp/frontend/dist \
    MONGODB_URI=mongodb://127.0.0.1:1 VTA_MAX_VIDEO_SECONDS=14400 VTA_RETENTION_DAYS=2 \
    YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-space.txt

COPY --chown=user . .

WORKDIR /home/user/app/webapp/backend
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
