FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to the container
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Set up entrypoint
RUN chmod +x start.sh
ENTRYPOINT ["./start.sh"]
