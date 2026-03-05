import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from slack_sdk import WebClient


class SlackMessageType(Enum):
    HEADER = 1
    DIVIDER = 2
    SECTION = 3
    TABLE = 4


@dataclass
class SlackMessage:
    msg_type: SlackMessageType = SlackMessageType.SECTION  # fixed type
    text: str = "default"


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Idempotent logger setup: no duplicate handlers, no global root mutations.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # keep formatting consistent, avoid double logs via root
    return logger


def format_slack_message(text: str, msg_type: SlackMessageType) -> Dict[str, Any]:
    if msg_type == SlackMessageType.HEADER:
        return {
            "type": "header",
            "text": {"type": "plain_text", "text": text, "emoji": False},
        }
    elif msg_type == SlackMessageType.SECTION:
        return {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
        }
    elif msg_type == SlackMessageType.DIVIDER:
        return {"type": "divider"}
    elif msg_type == SlackMessageType.TABLE:
        return {
            "type": "table",
            "column_settings": [{"is_wrapped": True}, {"align": "left"}],
            "rows": text,
        }
    else:
        raise NotImplementedError(f"Unsupported SlackMessageType: {msg_type}")


def send_slack_message(
    client: WebClient,
    channel_name: str,
    messages: Optional[List[SlackMessage]] = None,
    simple_text: Optional[str] = None,
    parent_message_ts: Optional[str] = None,
):
    """
    Posts either a simple text message or a set of block messages.
    Keeps your original behavior but a bit tighter.
    """
    if simple_text is not None:
        return client.chat_postMessage(
            channel=channel_name,
            text=simple_text,
            thread_ts=parent_message_ts,
        )

    blocks = [format_slack_message(text=m.text, msg_type=m.msg_type) for m in messages]
    return client.chat_postMessage(
        channel=channel_name,
        blocks=blocks,
        text="Hello",  # fallback text for notifications/search
        thread_ts=parent_message_ts,
    )
