FROM python:3.11-slim
# Hugging Face Spaces runs containers as uid 1000; create that user so caches/weights are writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501 \
    TORCH_HOME=/home/user/.cache/torch
WORKDIR /home/user/app
# CPU-only PyTorch keeps the image small
COPY --chown=user requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt
USER user
COPY --chown=user . .
# Pre-fetch the LPIPS (AlexNet) backbone at build time so the first request is not slow / offline-fragile.
RUN python -c "import lpips; lpips.LPIPS(net='alex', verbose=False)" || true
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
