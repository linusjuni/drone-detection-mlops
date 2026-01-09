"""Submit training job to Vertex AI."""

import os
from datetime import datetime
from google.cloud import aiplatform
import typer

from drone_detector_mlops.utils.logger import get_logger
from drone_detector_mlops.utils.settings import settings

logger = get_logger(__name__)
app = typer.Typer()


@app.command()
def main(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(0.001, help="Learning rate"),
    machine_type: str = typer.Option("n1-standard-4", help="Machine type"),
    accelerator_type: str = typer.Option("NVIDIA_TESLA_T4", help="GPU type (or 'None' for CPU)"),
    accelerator_count: int = typer.Option(1, help="Number of GPUs"),
    image_tag: str = typer.Option("latest", help="Docker image tag"),
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

    container_args = [
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
    ]

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
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    if not typer.confirm("Submit this job?", default=True):
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
                sync=True,  # Changed to True to see errors
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
                sync=True,  # Changed to True to see errors
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
        import traceback

        print("\n" + "=" * 80)
        print("FULL ERROR TRACEBACK:")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
