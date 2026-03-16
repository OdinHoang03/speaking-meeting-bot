import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger("meetingbaas-api")


def stringify_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: stringify_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [stringify_values(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def _freq_to_hz(freq: str) -> int:
    """Convert frequency string like '24khz' to integer Hz like 24000."""
    freq = freq.lower().strip()
    if freq.endswith("khz"):
        return int(freq.replace("khz", "")) * 1000
    try:
        return int(freq)
    except ValueError:
        return 24000


def create_meeting_bot(
    meeting_url: str,
    websocket_url: str,
    bot_id: str,
    persona_name: str,
    api_key: str,
    bot_image: Optional[str] = None,
    entry_message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    streaming_audio_frequency: str = "16khz",
    webhook_url: Optional[str] = None,
):
    if bot_image is not None:
        bot_image = str(bot_image)

    # Create the WebSocket path for streaming
    websocket_with_path = f"{websocket_url}/ws/{bot_id}"
    audio_hz = _freq_to_hz(streaming_audio_frequency)

    # Build v2 API request body
    config = {
        "meeting_url": meeting_url,
        "bot_name": persona_name,
        "reserved": False,
        "deduplication_key": f"{persona_name}-BaaS-{bot_id}",
        "streaming_enabled": True,
        "streaming_config": {
            "input_url": websocket_with_path,
            "output_url": websocket_with_path,
            "audio_frequency": audio_hz,
        },
    }

    if bot_image:
        config["bot_image"] = str(bot_image)
    if entry_message:
        config["entry_message"] = entry_message
    if extra:
        config["extra"] = extra
    if webhook_url:
        config["webhook_url"] = webhook_url

    config = stringify_values(config)

    url = "https://api.meetingbaas.com/v2/bots"
    headers = {
        "Content-Type": "application/json",
        "x-meeting-baas-api-key": api_key,
    }

    try:
        print(f"[MEETINGBAAS] API request: {url}")
        print(f"[MEETINGBAAS] streaming_config: input_url={config['streaming_config']['input_url']}, output_url={config['streaming_config']['output_url']}, audio_frequency={config['streaming_config']['audio_frequency']}")
        print(f"[MEETINGBAAS] Full payload: {json.dumps(config, indent=2)}")

        response = requests.post(url, json=config, headers=headers)
        print(f"[MEETINGBAAS] Response: {response.status_code} - {response.text[:500]}")

        if response.status_code in (200, 201):
            data = response.json()
            if "data" in data:
                bot_id = data["data"].get("bot_id") or data["data"].get("id")
            else:
                bot_id = data.get("bot_id")
            logger.info(f"Bot created with ID: {bot_id}")
            return bot_id
        else:
            logger.error(f"Failed to create bot: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error creating bot: {str(e)}")
        return None


def leave_meeting_bot(bot_id: str, api_key: str) -> bool:
    url = f"https://api.meetingbaas.com/v2/bots/{bot_id}"
    headers = {"x-meeting-baas-api-key": api_key}

    try:
        logger.info(f"Removing bot with ID: {bot_id}")
        response = requests.delete(url, headers=headers)
        if response.status_code == 200:
            logger.info(f"Bot {bot_id} successfully left the meeting")
            return True
        else:
            logger.error(f"Failed to remove bot: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error removing bot: {str(e)}")
        return False
