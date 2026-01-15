# Build stage
FROM node:22-slim AS builder

WORKDIR /app

# Default to localhost for local builds
ARG PUBLIC_API_URL=http://localhost:8080
ENV PUBLIC_API_URL=${PUBLIC_API_URL}

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY frontend/ ./

# Build the app (PUBLIC_API_URL is baked in here)
RUN npm run build

# Production stage
FROM node:22-slim AS production

WORKDIR /app

# Copy built app and production dependencies
COPY --from=builder /app/build ./build
COPY --from=builder /app/package*.json ./

# Install only production dependencies
RUN npm ci --omit=dev

ENV NODE_ENV=production
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "fetch('http://localhost:8080').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["node", "build"]
