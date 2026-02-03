FROM vllm/vllm-openai:v0.12.0

# Upgrade transformers (MedGemma 1.5 needs 4.50+) and install timm for vision backbone
RUN pip install --no-cache-dir --upgrade transformers "timm>=1.0.17"

# Install additional dependencies for RunPod worker
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH="/"

# Copy worker source code
COPY src /src

# Optional: bake model into image at build time
ARG MODEL_NAME=""
ARG MODEL_REVISION=""
ARG TOKENIZER_NAME=""
ARG TOKENIZER_REVISION=""
ARG QUANTIZATION=""
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    QUANTIZATION=$QUANTIZATION \
    BASE_PATH=$BASE_PATH \
    DTYPE=bfloat16 \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Reset the vLLM base image entrypoint so we can run our own handler
ENTRYPOINT []
CMD ["python3", "/src/handler.py"]
