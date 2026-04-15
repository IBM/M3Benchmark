FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY brand-logo.png vakra_capabilities_diagram.png image.png brand-mark.png app.py data.py leaderboard_data.json ui.html ./

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/ || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
