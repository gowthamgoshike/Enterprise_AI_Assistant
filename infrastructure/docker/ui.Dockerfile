FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "apps/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
