import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "drone_detector_mlops"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build training Docker image locally."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build API Docker image locally."""
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_all(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images locally."""
    docker_build_train(ctx, progress)
    docker_build_api(ctx, progress)


@task
def cloud_build_train(ctx: Context) -> None:
    """Build training image using Google Cloud Build."""
    ctx.run(
        "gcloud builds submit --config cloud/cloudbuild-train.yaml",
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_build_api(ctx: Context) -> None:
    """Build API image using Google Cloud Build."""
    ctx.run(
        "gcloud builds submit --config cloud/cloudbuild-api.yaml",
        echo=True,
        pty=not WINDOWS,
    )


@task
def cloud_build_all(ctx: Context) -> None:
    """Build all docker images using Google Cloud Build."""
    cloud_build_train(ctx)
    cloud_build_api(ctx)


@task
def cloud_train(
    ctx: Context,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
) -> None:
    """Submit training job to Vertex AI (n1-standard-4 + T4 GPU)."""
    cmd = (
        f"uv run -m scripts.submit_training "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
        f"--lr {lr} "
        f"--machine-type n1-standard-4 "
        f"--accelerator-type NVIDIA_TESLA_T4 "
        f"--accelerator-count 1 "
        f"--image-tag latest "
        f"--yes"
    )

    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def deploy_api(ctx: Context) -> None:
    """Deploy API to Cloud Run."""
    ctx.run(
        "gcloud run services update drone-detector-api "
        "--region europe-north2 "
        "--image europe-north2-docker.pkg.dev/drone-detection-mlops/ml-containers/api:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def load_test(ctx: Context, duration: int = 600, users: int = 10) -> None:
    """Run Locust load test in the browser."""
    ctx.run(
        "uv run locust -f tests/load/locustfile.py --host https://drone-detector-api-66108710596.europe-north2.run.app",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
