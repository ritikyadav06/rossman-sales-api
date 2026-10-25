FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
