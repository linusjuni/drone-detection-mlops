# Docker Containers

## Training Container

### Purpose

Containerized training environment for the drone detector model using multi-stage build with `uv` for dependency management.

### Build Process

1. **Stage 1**: Extract `uv` binary from official image
2. **Stage 2**:
   - Python 3.12 slim base based on Debian
   - Install some system dependencies (`libgl1`, `libglib2.0-0`)
   - Install Python packages via `uv`
   - Copy source code and data splits

### Usage

```bash
# Build
docker-compose build train

# Run with defaults
docker-compose up train

# Custom parameters
docker-compose run train --epochs 20 --batch-size 64 --lr 0.0001
```

### Container Structure

```plain
/app/
├── src/                   # Application code
├── data/splits/           # Train/val/test splits (built-in)
├── data/                  # Raw data (volume mounted)
└── models/                # Output (volume mounted)
```

## API Container

(Coming soon)
