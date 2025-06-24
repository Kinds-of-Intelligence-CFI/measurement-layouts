FROM python:3.10.12-slim

WORKDIR /app

# Install system deps (xorg/xvfb only if really needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    xvfb \
    x11-utils \
    xauth \ 
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your source code
COPY . .

# Optional: define default run command
# CMD ["python", "main.py"]
