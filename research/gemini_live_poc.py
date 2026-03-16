#!/usr/bin/env python3
"""
Gemini Live API — Proof of Concept

Demonstrates two approaches:
  1. Raw WebSocket connection to Gemini Live API (no pipecat dependency)
  2. Pipecat GeminiLiveLLMService integration

Usage:
  # Raw WebSocket PoC (default)
  python research/gemini_live_poc.py

  # Pipecat integration PoC
  python research/gemini_live_poc.py --pipecat

  # With custom prompt
  python research/gemini_live_poc.py --prompt "You are a pirate. Respond in pirate speak."

  # With a test audio file (16kHz mono PCM16 WAV)
  python research/gemini_live_poc.py --audio-file test.wav

Requirements:
  pip install websockets google-genai pipecat-ai[google]

Environment:
  GOOGLE_API_KEY=your_api_key
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import struct
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
DEFAULT_VOICE = "Kore"
DEFAULT_PROMPT = (
    "You are a friendly assistant in a brief audio test. "
    "When you hear audio input, respond with a short greeting. "
    "Keep responses under 2 sentences."
)
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
WS_URL_TEMPLATE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    "?key={api_key}"
)

# ============================================================================
# Approach 1: Raw WebSocket PoC
# ============================================================================


def generate_test_audio(duration_s: float = 1.0, freq_hz: float = 440.0) -> bytes:
    """Generate a sine wave tone as PCM16 audio for testing."""
    import math

    num_samples = int(INPUT_SAMPLE_RATE * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / INPUT_SAMPLE_RATE
        sample = int(16000 * math.sin(2 * math.pi * freq_hz * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, sample))))
    return b"".join(samples)


def load_audio_file(path: str) -> bytes:
    """Load a WAV file and return raw PCM16 bytes (strips header)."""
    import wave

    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit audio, got {wf.getsampwidth() * 8}-bit")
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono audio, got {wf.getnchannels()} channels")
        rate = wf.getframerate()
        if rate != INPUT_SAMPLE_RATE:
            logger.warning(
                f"Audio file is {rate}Hz, expected {INPUT_SAMPLE_RATE}Hz. "
                "Sending as-is — Gemini may handle it but quality may vary."
            )
        return wf.readframes(wf.getnframes())


async def raw_websocket_poc(
    api_key: str,
    system_instruction: str,
    audio_data: bytes | None = None,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
):
    """Connect to Gemini Live via raw WebSocket and demonstrate basic flow."""
    try:
        import websockets
    except ImportError:
        logger.error("Install websockets: pip install websockets")
        sys.exit(1)

    url = WS_URL_TEMPLATE.format(api_key=api_key)
    logger.info(f"Connecting to Gemini Live API (model={model}, voice={voice})...")

    async with websockets.connect(url) as ws:
        # Step 1: Send setup message
        setup_msg = {
            "setup": {
                "model": model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": voice}
                        }
                    },
                },
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        "startOfSpeechSensitivity": "START_SENSITIVITY_LOW",
                        "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
                        "prefixPaddingMs": 20,
                        "silenceDurationMs": 500,
                    }
                },
            }
        }
        await ws.send(json.dumps(setup_msg))
        logger.info("Sent setup message")

        # Step 2: Wait for setupComplete
        response = json.loads(await ws.recv())
        if "setupComplete" in response:
            logger.info("Session established successfully!")
        else:
            logger.error(f"Unexpected response: {response}")
            return

        # Step 3: Send audio (test tone or file)
        if audio_data is None:
            logger.info("Generating 1-second test tone (440Hz)...")
            audio_data = generate_test_audio(duration_s=1.0)

        # Send audio in chunks (~100ms each)
        chunk_size = INPUT_SAMPLE_RATE * 2 // 10  # 100ms of PCM16
        total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        logger.info(
            f"Streaming {len(audio_data)} bytes of audio "
            f"({len(audio_data) / (INPUT_SAMPLE_RATE * 2):.1f}s, {total_chunks} chunks)..."
        )

        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            audio_msg = {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(chunk).decode("utf-8"),
                        "mimeType": f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    }
                }
            }
            await ws.send(json.dumps(audio_msg))

        # Send a small silence gap to trigger end-of-speech
        silence = b"\x00" * (INPUT_SAMPLE_RATE * 2)  # 1s silence
        await ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(silence).decode("utf-8"),
                            "mimeType": f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                        }
                    }
                }
            )
        )
        logger.info("Audio sent. Waiting for response...")

        # Step 4: Collect response
        output_audio = bytearray()
        transcript_parts = []
        start_time = time.time()
        first_audio_time = None

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                logger.info("No more responses (10s timeout)")
                break

            if "serverContent" in msg:
                sc = msg["serverContent"]

                # Collect audio from model turn
                if "modelTurn" in sc:
                    for part in sc["modelTurn"].get("parts", []):
                        if "inlineData" in part:
                            audio_bytes = base64.b64decode(part["inlineData"]["data"])
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                latency = first_audio_time - start_time
                                logger.info(f"First audio received! Latency: {latency:.2f}s")
                            output_audio.extend(audio_bytes)

                # Collect transcriptions
                if "outputTranscription" in sc:
                    text = sc["outputTranscription"].get("text", "")
                    if text:
                        transcript_parts.append(text)

                if "inputTranscription" in sc:
                    text = sc["inputTranscription"].get("text", "")
                    if text:
                        logger.info(f"Input transcription: {text}")

                # Check for turn completion
                if sc.get("turnComplete"):
                    logger.info("Model turn complete")
                    break

                if sc.get("interrupted"):
                    logger.info("Model was interrupted")
                    break

            elif "usageMetadata" in msg:
                meta = msg["usageMetadata"]
                logger.info(f"Token usage: {json.dumps(meta, indent=2)}")

        # Summary
        duration = len(output_audio) / (OUTPUT_SAMPLE_RATE * 2) if output_audio else 0
        transcript = "".join(transcript_parts)

        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Output audio: {len(output_audio)} bytes ({duration:.1f}s at {OUTPUT_SAMPLE_RATE}Hz)")
        if transcript:
            logger.info(f"Transcript: {transcript}")
        else:
            logger.info("Transcript: (none received — enable outputAudioTranscription for text)")
        if first_audio_time:
            logger.info(f"Time to first audio: {first_audio_time - start_time:.2f}s")

        # Optionally save output audio
        if output_audio:
            output_path = "research/gemini_live_output.raw"
            with open(output_path, "wb") as f:
                f.write(output_audio)
            logger.info(f"Raw output saved to {output_path}")
            logger.info(f"Play with: aplay -f S16_LE -r {OUTPUT_SAMPLE_RATE} -c 1 {output_path}")

        return len(output_audio) > 0


# ============================================================================
# Approach 2: Pipecat Integration PoC
# ============================================================================


async def pipecat_poc(
    api_key: str,
    system_instruction: str,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
):
    """Demonstrate GeminiLiveLLMService usage in a minimal pipecat pipeline."""
    try:
        from pipecat.services.google.gemini_live import GeminiLiveLLMService
    except ImportError:
        logger.error(
            "GeminiLiveLLMService not available. "
            "Install with: pip install 'pipecat-ai[google]'"
        )
        sys.exit(1)

    logger.info("Pipecat GeminiLiveLLMService is available!")
    logger.info(f"Module: {GeminiLiveLLMService.__module__}")

    # Show the service can be instantiated
    try:
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMSettings

        settings = GeminiLiveLLMSettings(
            model=model,
            voice=voice,
        )
        logger.info(f"Settings created: model={settings.model}, voice={settings.voice}")

        service = GeminiLiveLLMService(
            api_key=api_key,
            system_instruction=system_instruction,
            settings=settings,
        )
        logger.info(f"GeminiLiveLLMService instantiated successfully")
        logger.info(f"  Sample rate: {service.sample_rate}")
        logger.info(f"  Service name: {service.name}")

        # Show example pipeline structure
        logger.info("")
        logger.info("Example pipeline for meetingbaas.py integration:")
        logger.info("=" * 60)
        logger.info("""
from pipecat.services.google.gemini_live import GeminiLiveLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMSettings

gemini_live = GeminiLiveLLMService(
    api_key=os.getenv("GOOGLE_API_KEY"),
    system_instruction=persona["prompt"],
    settings=GeminiLiveLLMSettings(
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        voice=persona.get("gemini_live_voice", "Kore"),
    ),
)

pipeline = Pipeline([
    transport.input(),
    context_aggregator.user(),
    gemini_live,
    transport.output(),
    context_aggregator.assistant(),
])
""")

    except Exception as e:
        logger.error(f"Failed to instantiate service: {e}")
        raise

    return True


# ============================================================================
# Main
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Gemini Live API Proof of Concept")
    parser.add_argument(
        "--pipecat",
        action="store_true",
        help="Run the pipecat integration PoC instead of raw WebSocket",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="System instruction / persona prompt",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Voice name (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--audio-file",
        help="Path to a 16kHz mono PCM16 WAV file to send as input",
    )
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Set GOOGLE_API_KEY environment variable")
        sys.exit(1)

    audio_data = None
    if args.audio_file:
        audio_data = load_audio_file(args.audio_file)
        logger.info(f"Loaded audio file: {args.audio_file} ({len(audio_data)} bytes)")

    if args.pipecat:
        success = await pipecat_poc(
            api_key=api_key,
            system_instruction=args.prompt,
            model=args.model,
            voice=args.voice,
        )
    else:
        success = await raw_websocket_poc(
            api_key=api_key,
            system_instruction=args.prompt,
            audio_data=audio_data,
            model=args.model,
            voice=args.voice,
        )

    if success:
        logger.info("PoC completed successfully!")
    else:
        logger.warning("PoC completed but no audio response received")


if __name__ == "__main__":
    asyncio.run(main())
