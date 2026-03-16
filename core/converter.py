"""Handles conversion between raw audio and Protobuf frames."""

from typing import Optional

import protobufs.frames_pb2 as frames_pb2
from meetingbaas_pipecat.utils.logger import logger


class ProtobufConverter:
    """Handles conversion between raw audio and Protobuf frames."""

    def __init__(self, logger=logger, sample_rate: int = 24000, channels: int = 1):
        self.logger = logger
        self.sample_rate = sample_rate
        self.channels = channels

    def set_sample_rate(self, sample_rate: int):
        """Update the sample rate."""
        self.sample_rate = sample_rate
        self.logger.info(f"Updated ProtobufConverter sample rate to {sample_rate}")

    def raw_to_protobuf(self, raw_audio: bytes) -> bytes:
        """Convert raw audio data to a serialized Protobuf frame."""
        try:
            frame = frames_pb2.Frame()
            frame.audio.audio = raw_audio
            frame.audio.sample_rate = self.sample_rate
            frame.audio.num_channels = self.channels

            return frame.SerializeToString()
        except Exception as e:
            self.logger.error(f"Error converting raw audio to Protobuf: {str(e)}")
            raise

    def protobuf_to_raw(self, proto_data: bytes) -> Optional[bytes]:
        """Extract raw audio from a serialized Protobuf frame."""
        try:
            frame = frames_pb2.Frame()
            frame.ParseFromString(proto_data)

            if frame.HasField("audio"):
                return bytes(frame.audio.audio)
            return None
        except Exception as e:
            self.logger.error(f"Error extracting audio from Protobuf: {str(e)}")
            return None


# Create a singleton instance
converter = ProtobufConverter()
