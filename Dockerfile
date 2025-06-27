FROM python:3.10.9-slim

WORKDIR /app

COPY . .

# Instalar pip y dependencias
RUN apt-get update && apt-get install -y python3-pip

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
