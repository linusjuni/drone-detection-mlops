import os
from datetime import datetime
from google.cloud import aiplatform
import typer
import traceback

from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.settings import settings

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    machine_type: str = typer.Option("n1-standard-4", help="Machine type"),
    accelerator_type: str = typer.Option("NVIDIA_TESLA_T4", help="GPU type (or 'None' for CPU)"),
    accelerator_count: int = typer.Option(1, help="Number of GPUs"),
    image_tag: str = typer.Option("latest", help="Docker image tag"),
    sweep: bool = typer.Option(False, help="Run Optuna sweep"),
    hydra_overrides: str = typer.Option("", help="Hydra overrides (comma-separated)"),
    config: str = typer.Option("param_1", help="Hyperparameter config (param_1, param_2)"),
    epochs: int = typer.Option(None, help="Override epochs"),
    batch_size: int = typer.Option(None, help="Override batch size"),
    lr: float = typer.Option(None, help="Override learning rate"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Submit a training job to Vertex AI."""

    PROJECT_ID = settings.GCP_PROJECT
    REGION = settings.GCP_REGION
    STAGING_BUCKET = getattr(settings, "GCS_STAGING_BUCKET", "gs://drone-detection-mlops-staging-west4/")
    IMAGE_URI = f"europe-north2-docker.pkg.dev/{PROJECT_ID}/ml-containers/train:{image_tag}"

    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        logger.error("WANDB_API_KEY not found in environment")
        logger.info("Set it with: export WANDB_API_KEY=your_key")
        raise typer.Exit(1)

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"drone-detector-{timestamp}"

    # Build container args based on options
    container_args = []
    if sweep:
        container_args.extend(["--multirun", "+sweep=basic"])
    elif hydra_overrides:
        # Custom Hydra overrides
        container_args.extend(hydra_overrides.split(","))
    else:
        # Use specified config
        container_args.append(f"hyper_parameters={config}")

        # Apply individual overrides if provided
        if epochs is not None:
            container_args.append(f"hyper_parameters.epochs={epochs}")
        if batch_size is not None:
            container_args.append(f"hyper_parameters.batch_size={batch_size}")
        if lr is not None:
            container_args.append(f"hyper_parameters.lr={lr}")

    env_vars = {
        "MODE": "cloud",
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT_NAME": settings.WANDB_PROJECT_NAME,
        "GCS_DATA_PATH": settings.GCS_DATA_PATH,
        "GCS_MODELS_BUCKET": settings.GCS_MODELS_BUCKET,
        "GCP_PROJECT": settings.GCP_PROJECT,
        "GCP_REGION": settings.GCP_REGION,
    }

    logger.info(
        "Submitting Vertex AI training job",
        display_name=display_name,
        image=IMAGE_URI,
        machine_type=machine_type,
        accelerator=accelerator_type if accelerator_type != "None" else "CPU",
        sweep=sweep,
        container_args=container_args,
    )

    if not yes and not typer.confirm("Submit this job?", default=True):
        logger.warning("Job submission cancelled")
        raise typer.Exit(0)

    try:
        logger.info("Creating CustomContainerTrainingJob...")
        job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=IMAGE_URI,
        )

        logger.info("Job object created, submitting to Vertex AI...")

        if accelerator_type == "None":
            logger.info("Submitting CPU job...")
            result = job.run(
                args=container_args,
                environment_variables=env_vars,
                replica_count=1,
                machine_type=machine_type,
                sync=True,
            )
        else:
            logger.info("Submitting GPU job...")
            result = job.run(
                args=container_args,
                environment_variables=env_vars,
                replica_count=1,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                sync=True,
            )

        logger.info("Job.run() returned", result=str(result))
        logger.success("Job submitted successfully")

        # Try to get resource name if available
        try:
            if hasattr(job, "_gca_resource") and job._gca_resource:
                logger.info("Job resource", name=job.resource_name)
        except Exception:
            logger.info("Job submitted", display_name=display_name)

        logger.info(
            "Monitor job",
            console_url=f"https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}",
        )
        logger.info(
            "View logs",
            logs_url=f"https://console.cloud.google.com/logs/query?project={PROJECT_ID}",
        )

    except Exception as e:
        logger.error("Job submission failed", error=str(e))

        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
