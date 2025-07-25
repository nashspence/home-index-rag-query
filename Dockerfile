FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "app/main.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
