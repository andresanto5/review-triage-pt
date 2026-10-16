FROM python:3.12-slim
WORKDIR /project
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV MLFLOW_ENABLE_TELEMETRY=false DO_NOT_TRACK=true
CMD ["python", "src/analyze.py"]
