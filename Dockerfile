FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

# Install build tools and any additional dependencies
RUN apt-get update && \
apt-get upgrade -y && \
apt-get install -y --no-install-recommends \
build-essential cmake git libopencv-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./src /app
COPY ./models--bert-base-uncased /root/.cache/huggingface/hub/models--bert-base-uncased

COPY ./req.txt /app/req.txt
RUN pip3 install --no-cache-dir -r req.txt

# Expose port 8080 for the FastAPI app
EXPOSE 8080

# ENTRYPOINT [ "python3", "sam2_prompt_defect_detection.py" ]
ENTRYPOINT [ "uvicorn", "sam2_prompt_defect_detection:app" , "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]

