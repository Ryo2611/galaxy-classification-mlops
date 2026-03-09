# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
# Add web framework dependencies
RUN echo "fastapi[all]" >> /app/requirements.txt
RUN echo "streamlit" >> /app/requirements.txt
RUN echo "requests" >> /app/requirements.txt
RUN echo "python-multipart" >> /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY ./app /app/app
COPY ./src /app/src
COPY ./models /app/models

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501
