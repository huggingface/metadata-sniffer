from dotenv import load_dotenv

load_dotenv()
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, ModelInfo
from typing import List
from slack_sdk import WebClient
from configuration import SlackConfig
from utilities import SlackMessage, send_slack_message, setup_logging, SlackMessageType
from dataclasses import dataclass, asdict

logger = setup_logging(__name__)
from datasets import Dataset
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
today = now.strftime("%Y-%m-%d")


@dataclass
class CustomCodeResult:
    id: str
    days_passed: int
    num_downloads: int


def analyze_custom_model_metadata(
    huggingface_api: HfApi,
    model_info: ModelInfo,
) -> CustomCodeResult:
    model_id = model_info.id
    num_downloads = model_info.downloads
    days_passed = (now - model_info.created_at).days
    custom_code_result = CustomCodeResult(
        id=model_id, days_passed=days_passed, num_downloads=num_downloads
    )
    return custom_code_result


if __name__ == "__main__":
    # configuration
    hf_token = os.environ["HF_TOKEN"]
    slack_token = os.environ["SLACK_TOKEN"]
    huggingface_api = HfApi(token=hf_token)
    slack_client = WebClient(token=slack_token)
    slack_config = SlackConfig()

    custom_code_models = huggingface_api.list_models(
        filter=["custom_code", "transformers"],
        limit=100,
        token=hf_token,
        full=True,
    )

    custom_code_results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model_info = {
            executor.submit(
                analyze_custom_model_metadata,
                huggingface_api,
                model_info,
            ): model_info
            for model_info in custom_code_models
        }

        for future in as_completed(future_to_model_info):
            model_info = future_to_model_info[future]
            try:
                result = future.result()
                custom_code_results.append(asdict(result))
            except Exception as e:
                logger.error(f"Error processing model {model_info.id}: {e}")

    trending_custom_models_ds = Dataset.from_list(custom_code_results)
    trending_custom_models_ds_sorted = trending_custom_models_ds.sort(
        column_names="num_downloads",
        reverse=True,
    )
    column_names = trending_custom_models_ds_sorted.column_names
    rows = [[{"type": "raw_text", "text": column_name} for column_name in column_names]]
    for row in trending_custom_models_ds_sorted.take(10):
        single_row = []
        for key, value in row.items():
            if key == "id":
                single_row.append(
                    {
                        "type": "rich_text",
                        "elements": [
                            {
                                "type": "rich_text_section",
                                "elements": [
                                    {
                                        "text": f"{value}",
                                        "type": "link",
                                        "url": f"https://huggingface.co/{value}",
                                    }
                                ],
                            }
                        ],
                    }
                )
            else:
                single_row.append(
                    {
                        "type": "raw_text",
                        "text": f"{value}",
                    }
                )

        rows.append(single_row)
    trending_custom_models_ds.push_to_hub(
        repo_id="model-metadata/custom-code-models", token=hf_token
    )

    messages = [
        SlackMessage(msg_type=SlackMessageType.DIVIDER),
        SlackMessage(
            text=f"Custom Transformers Code {today}",
            msg_type=SlackMessageType.HEADER,
        ),
        SlackMessage(msg_type=SlackMessageType.TABLE, text=rows),
        SlackMessage(
            text=f"Rest is uploaded to: <https://huggingface.co/datasets/model-metadata/custom-code-models|dataset>"
        ),
    ]

    send_slack_message(
        client=slack_client, channel_name=slack_config.channel_name, messages=messages
    )
