import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict

from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, ModelInfo
from slack_sdk import WebClient

from configuration import ModelCheckerConfig, SlackConfig, DatasetConfig
from utilities import SlackMessage, SlackMessageType, send_slack_message, setup_logging
from datetime import datetime, timezone

load_dotenv()
logger = setup_logging(__name__)

now = datetime.now(timezone.utc)
today = now.strftime("%Y-%m-%d")


@dataclass
class OpenAvocadoDiscussion:
    title: str
    author: str
    url: str
    status: str
    created_at: datetime
    days_passed: int


@dataclass
class ModelMetadataResult:
    id: str
    should_skip_code_exec: bool = False
    metadata_issues: List[str] = field(default_factory=list)
    open_discussions_with_avocado_participation: List[OpenAvocadoDiscussion] = field(
        default_factory=list
    )


class MetadataIssues(Enum):
    NO_LIBRARY_NAME = "no_library_name"
    NO_PIPELINE_TAG = "no_pipeline_tag"
    HAS_NOTEBOOK = "has_notebook_needs_manual_execution"


def _model_link_line(model_id) -> str:
    return f"* <https://huggingface.co/{model_id}|{model_id}>\n"


def _discussion_link_line(model_id, discussions) -> str:
    text = f"* {model_id}: "
    for discussion in discussions:
        days_passed = discussion["days_passed"]
        text = text + f" <{discussion['url']}|{days_passed}-days-old>"
    text = text + "\n"
    return text


def _chunk_markdown(text_lines: List[str], max_len: int = 2900) -> List[str]:
    """Split lines into chunks under Slack section hard-limit."""
    chunks: List[str] = []
    buf = ""
    for line in text_lines:
        if len(buf) + len(line) > max_len:
            chunks.append(buf)
            buf = line
        else:
            buf += line
    if buf:
        chunks.append(buf)
    return chunks


def analyze_model_metadata(
    huggingface_api: HfApi,
    model_info: ModelInfo,
    avocado_team_members: List[str],
) -> ModelMetadataResult:
    """Analyze metadata for a model"""
    model_id = model_info.id
    metadata_result = ModelMetadataResult(id=model_id)

    # we will currently ignore GGUFs
    if "gguf" in (model_info.tags or []):
        metadata_result.should_skip_code_exec = True
        logger.info(f"Skipped {model_id} : GGUF")
        return metadata_result

    # some models do not have a discussion tab, we will ignore them
    try:
        discussions = list(huggingface_api.get_repo_discussions(model_id))
    except Exception:
        metadata_result.should_skip_code_exec = True
        metadata_result.has_notebook = True
        logger.info(f"Skipped {model_id} : No Discussion Tab")
        return metadata_result

    # with open avocado discussions we want to do two things
    # 1. let the team know if we have a discussion open for 3 days
    # 2. if there are merged PRs but the model still has issues, we will still alert the team
    open_discussions_with_avocado: List[OpenAvocadoDiscussion] = []
    for discussion in discussions:
        if (
            discussion.author in avocado_team_members and discussion.status == "open"
        ):  # check if one of the avocado team memeber has interacted with the model and is still open
            days_passed = (now - discussion.created_at).days
            open_discussions_with_avocado.append(
                OpenAvocadoDiscussion(
                    title=discussion.title,
                    status=discussion.status,
                    created_at=discussion.created_at,
                    author=discussion.author,
                    url=f"https://huggingface.co/{model_id}/discussions/{discussion.num}",
                    days_passed=days_passed,
                )
            )
    metadata_result.open_discussions_with_avocado_participation = (
        open_discussions_with_avocado
    )

    # Metadata issues
    if model_info.library_name is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_LIBRARY_NAME.value)

    if model_info.pipeline_tag is None:
        metadata_result.metadata_issues.append(MetadataIssues.NO_PIPELINE_TAG.value)

    # check for notebook present
    if "notebook.ipynb" in [f.rfilename for f in model_info.siblings]:
        metadata_result.should_skip_code_exec = True
        metadata_result.metadata_issues.append(MetadataIssues.HAS_NOTEBOOK.value)
        logger.info(f"Notebook found in {model_id}")

    return metadata_result


if __name__ == "__main__":
    # configuration
    hf_token = os.environ["HF_TOKEN"]
    slack_token = os.environ["SLACK_TOKEN"]
    huggingface_api = HfApi(token=hf_token)
    slack_client = WebClient(token=slack_token)
    dataset_config = DatasetConfig()
    slack_config = SlackConfig()
    model_checker_config = ModelCheckerConfig()

    # fetch the top N trending models
    trending_models = huggingface_api.list_models(
        sort="trendingScore",
        limit=model_checker_config.num_trending_models,
        token=hf_token,
        full=True,  # need to get the repo siblings
    )

    # process model metadata
    avocado_members = list(getattr(model_checker_config, "avocado_team_members", []))
    metadata_results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_model_info = {
            executor.submit(
                analyze_model_metadata, huggingface_api, model_info, avocado_members
            ): model_info
            for model_info in trending_models
        }

        for future in as_completed(future_to_model_info):
            model_info = future_to_model_info[future]
            try:
                result = future.result()
                metadata_results.append(asdict(result))  # keep Dataset conversion
            except Exception as e:
                logger.error(f"Error processing model {model_info.id}: {e}")

    trending_models_metadata_ds = Dataset.from_list(metadata_results)

    # categorize the models by issue to be alerted to the team
    # also collect the pending discussions
    models_by_issue_type: Dict[str, List[Dict]] = {
        issue.value: [] for issue in MetadataIssues
    }
    models_with_open_pending_prs: List[Dict] = []
    for row in trending_models_metadata_ds:
        open_avocado_discussions = row["open_discussions_with_avocado_participation"]
        if open_avocado_discussions:
            # do not add model id to problems (as this is already been taken care of)
            models_with_open_pending_prs.append(
                {"id": row["id"], "open_discussions": open_avocado_discussions}
            )
            continue

        for issue in row["metadata_issues"]:
            models_by_issue_type[issue].append(row["id"])

    # send the updates to slack
    messages = [
        SlackMessage(msg_type=SlackMessageType.DIVIDER),
        SlackMessage(
            text=f"Meta Data Report for {today}", msg_type=SlackMessageType.HEADER
        ),
    ]
    send_slack_message(
        client=slack_client, channel_name=slack_config.channel_name, messages=messages
    )

    # alert slack with the issues
    for issue_type, models in models_by_issue_type.items():
        title_msg = SlackMessage(
            text=f"*{' '.join(issue_type.split('_'))}*",
            msg_type=SlackMessageType.SECTION,
        )

        if not models:
            # no issues found
            send_slack_message(
                client=slack_client,
                channel_name=slack_config.channel_name,
                messages=[
                    title_msg,
                    SlackMessage(
                        text="Nothing today 🤙", msg_type=SlackMessageType.SECTION
                    ),
                ],
            )
            continue

        lines = [_model_link_line(model) for model in models]
        for chunk in _chunk_markdown(lines, max_len=2900):
            send_slack_message(
                client=slack_client,
                channel_name=slack_config.channel_name,
                messages=[
                    title_msg,
                    SlackMessage(text=chunk, msg_type=SlackMessageType.SECTION),
                ],
            )

    # alert for the pending discussions
    title_msg = SlackMessage(
        text=f"*Pending Discussions*",
        msg_type=SlackMessageType.SECTION,
    )
    lines = [
        _discussion_link_line(
            model_id=sample["id"], discussions=sample["open_discussions"]
        )
        for sample in models_with_open_pending_prs
    ]
    for chunk in _chunk_markdown(lines, max_len=2900):
        send_slack_message(
            client=slack_client,
            channel_name=slack_config.channel_name,
            messages=[
                title_msg,
                SlackMessage(text=chunk, msg_type=SlackMessageType.SECTION),
            ],
        )

    # Push the model dataset to Hub
    trending_models_metadata_ds.push_to_hub(
        dataset_config.trending_models_metadata_id, token=hf_token
    )
    send_slack_message(
        client=slack_client,
        channel_name=slack_config.channel_name,
        simple_text=f"Trending Models Dataset Uploaded to <https://huggingface.co/datasets/{dataset_config.trending_models_metadata_id}|{dataset_config.trending_models_metadata_id}>",
    )
