# ─────────────────────────────────────────────────
# Stage 1: Build React/Vite Frontend
# ─────────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy all frontend source and build
COPY frontend/ ./
RUN npm run build
# Output → /app/frontend/dist


# ─────────────────────────────────────────────────
# Stage 2: Python backend + serve frontend
# ─────────────────────────────────────────────────
FROM python:3.11-slim

# Install system packages: supervisor to run both services, nodejs+npm to run serve
RUN apt-get update && apt-get install -y \
    supervisor \
    nodejs \
    npm \
    && npm install -g serve \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy entire backend (includes model/ and watch_folder/)
COPY backend/ ./backend/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Flask API on 5000, React frontend on 5173
EXPOSE 5000 5173

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]