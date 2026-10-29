# Speech-to-speech orchestrator: LiveKit agent worker + demo web app (one image,
# two compose services). No GPU: it routes audio and text between the model
# services and runs voice activity detection (Silero, CPU) on each microphone.
FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
# silero-vad only for its ONNX model file (its torch dependency isn't needed)
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir --no-deps silero-vad==6.2.3

WORKDIR /srv
# End-of-turn model (app/turn.py): Pipecat's Smart Turn v3.2, CPU build, 8 MB, BSD-2-Clause;
# pinned to a Hugging Face revision and checked against its hash
ADD --checksum=sha256:2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f \
    https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx \
    /srv/models/smart-turn-v3.2-cpu.onnx
COPY app /srv/app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
