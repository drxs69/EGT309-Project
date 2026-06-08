# Dockerfile
# ----------
# Containerised environment for the ElderGuard Analytics ML pipeline.
# Provides a reproducible Python runtime with all required dependencies.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure the data and output directories exist
RUN mkdir -p data saved_model

# Default command: run the pipeline
CMD ["python", "src/pipeline.py"]
