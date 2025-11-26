# Multi-stage Dockerfile for Hugging Face Spaces
# This Dockerfile builds the React frontend and serves it alongside the Flask backend
# It automatically fixes the localhost:5000 URLs to work in the Space environment

# ---------------------------------------
# Stage 1: Build Frontend
# ---------------------------------------
FROM node:18-alpine as frontend-build
WORKDIR /app/frontend

# Install dependencies
COPY frontend/package*.json ./
RUN npm ci

# Copy source code
COPY frontend/ .

# MAGIC FIX: Replace hardcoded localhost:5000 with empty string (relative paths)
# This allows the frontend to talk to the backend on the same domain/port via Nginx
RUN sed -i 's|http://localhost:5000||g' src/App.tsx
RUN sed -i 's|http://localhost:5000||g' src/components/StatusPage.tsx

# Build React app
RUN npm run build

# ---------------------------------------
# Stage 2: Production Image (Python + Nginx)
# ---------------------------------------
FROM python:3.10-slim
WORKDIR /app

# Install Nginx and system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    nginx \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend Code
COPY backend/ .

# Copy Frontend Build from Stage 1
COPY --from=frontend-build /app/frontend/build /app/frontend_build

# Configure Nginx
# We configure it to listen on port 7860 (Hugging Face default)
# It serves the React app static files
# And proxies API requests to the Flask backend on port 5000
RUN echo 'server { \
    listen 7860; \
    server_name localhost; \
    \
    # Increase body size for video uploads \
    client_max_body_size 100M; \
    \
    location / { \
    root /app/frontend_build; \
    index index.html; \
    try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy API endpoints to Flask \
    location /analyze { \
    proxy_pass http://127.0.0.1:5000; \
    proxy_read_timeout 300; \
    } \
    location /upload { \
    proxy_pass http://127.0.0.1:5000; \
    } \
    location /health { \
    proxy_pass http://127.0.0.1:5000; \
    } \
    location /debug-model { \
    proxy_pass http://127.0.0.1:5000; \
    proxy_read_timeout 300; \
    } \
    }' > /etc/nginx/sites-available/default

# Create model cache directory with permissions
RUN mkdir -p model_cache && chmod 777 model_cache

# Create startup script
RUN echo '#!/bin/bash\n\
    # Start Flask in background\n\
    python app.py &\n\
    # Start Nginx in foreground\n\
    nginx -g "daemon off;"' > /app/start.sh && chmod +x /app/start.sh

# Expose Hugging Face Space port
EXPOSE 7860

# Run the startup script
CMD ["/app/start.sh"]
