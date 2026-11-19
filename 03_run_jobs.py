from huggingface_hub import run_job
import os
from dotenv import load_dotenv
from configuration import DatasetConfig, ExecuteCodeConfig, SlackConfig
from utilities import setup_logging
from datasets import load_dataset

load_dotenv()
logger = setup_logging(__name__)

GPU_VRAM_MAPPING = {
    "l4x1": 30,
    "a10g-large": 46,
}
VRAM_TO_GPU_MAPPING = dict(
    sorted({vram: gpu for gpu, vram in GPU_VRAM_MAPPING.items()}.items())
)


def select_appropriate_gpu(estimated_vram: float, model_id: str):
    for available_vram, gpu_type in VRAM_TO_GPU_MAPPING.items():
        if estimated_vram < available_vram * 0.9:  # Leave 10% headroom
            return gpu_type
    logger.warning(
        f"No suitable GPU found for {model_id} with {estimated_vram:.2f} GB VRAM requirement"
    )
    return None


if __name__ == "__main__":
    hf_token = os.environ["HF_TOKEN"]
    slack_token = os.environ["SLACK_TOKEN"]
    execution_config = ExecuteCodeConfig()
    docker_image = execution_config.docker_image
    datasets_config = DatasetConfig()
    slack_config = SlackConfig()

    hf_jobs_url_ds = load_dataset(datasets_config.hf_jobs_url_dataset_id, split="train")

    for row in hf_jobs_url_ds:
        selected_gpu = select_appropriate_gpu(
            estimated_vram=row["estimated_vram"], model_id=row["id"]
        )
        if selected_gpu is not None:
            for script_url in row["code_urls"]:
                run_job(
                    namespace=execution_config.jobs_namespace,
                    token=hf_token,
                    image=docker_image,
                    command=[
                        "/bin/bash",
                        "-c",
                        f"export HOME=/tmp && export USER=dummy && uv run {script_url}",
                    ],
                    flavor=selected_gpu,
                    secrets={
                        "HF_TOKEN": hf_token,
                        "SLACK_TOKEN": slack_token,
                    },
                )
