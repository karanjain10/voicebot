FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN python train.py
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "app:app"]
