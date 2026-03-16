# Gemini Live API & Pipecat Compatibility — Research Findings

## Executive Summary

**Gemini Live API** (Google's Multimodal Live API) provides real-time, bidirectional audio streaming via WebSocket — replacing the traditional STT → LLM → TTS pipeline with a single speech-to-speech connection.

**Key finding:** pipecat-ai v0.0.105 (our current version) already ships `GeminiLiveLLMService`, a mature, production-ready wrapper. Integration requires minimal code changes — primarily pipeline restructuring and a new provider path.

---

## 1. Gemini Live API Capabilities

### Protocol
- **Transport:** WebSocket (bidirectional JSON messages)
- **URL:** `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}`
- **Flow:** Client sends `setup` → server responds `setupComplete` → client streams `realtimeInput` (audio chunks) → server streams `serverContent` (audio responses)

### Audio Formats

| Direction | Format | Sample Rate | Channels | MIME Type |
|-----------|--------|-------------|----------|-----------|
| Input     | PCM16 LE | **16 kHz** | Mono | `audio/pcm;rate=16000` |
| Output    | PCM16 LE | **24 kHz** | Mono | `audio/pcm;rate=24000` |

Audio is base64-encoded in the JSON messages.

### System Instructions & Persona
- Set via `setup.systemInstruction.parts[].text` — standard Gemini `Content` format
- Text-only (no audio/image system instructions)
- Preserved during context window compression (sliding window)
- Supports multi-paragraph instructions — our persona prompts map directly

### Interruption Handling
- **Built-in VAD** with configurable sensitivity (`startOfSpeechSensitivity`, `endOfSpeechSensitivity`)
- **Barge-in:** When user speaks during model output, generation is cancelled; server sends `interrupted: true`
- **Manual VAD mode:** Disable auto-VAD and send `activityStart`/`activityEnd` signals manually
- **Activity handling:** `START_OF_ACTIVITY_INTERRUPTS` (default) or `NO_INTERRUPTION`

### Session Management
- **Duration:** ~15 min audio-only, ~2 min with video (without compression)
- **WebSocket lifetime:** ~10 min max (use session resumption)
- **Session resumption:** Server provides tokens valid for 2 hours; reconnect with last token
- **Context compression:** Sliding window available, preserves system instructions

### Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| `gemini-2.5-flash-native-audio-preview-12-2025` | Preview (Recommended) | Native audio, sub-second latency, 131K context |
| `gemini-2.5-flash-native-audio-latest` | Preview | Alias to latest |
| `gemini-2.5-flash-preview-native-audio-dialog` | Preview | Dialog-optimized |
| `gemini-2.5-flash-exp-native-audio-thinking-dialog` | Experimental | With reasoning |

Native audio models skip separate STT/TTS — the model processes audio directly, yielding lower latency.

### Available Voices
- **Core:** Aoede, Charon, Fenrir, Kore, Puck
- **Extended (native audio):** Leda, Orus, Zephyr + 20+ more HD voices
- 24 languages, 70 input languages

### Pricing (Paid Tier)

| Component | Cost |
|-----------|------|
| Audio Input | $3.00 / million tokens |
| Audio Output | $12.00 / million tokens |

Free tier available with lower rate limits.

### Additional Features
- **Input/output transcription** — get text of what user said and what model said
- **Function calling** — model can request tool execution mid-conversation
- **Thinking/reasoning** — configurable budget for chain-of-thought
- **Proactive audio** — model decides when to respond vs stay silent
- **Affective dialog** — adapts tone to emotional context (v1alpha API)

---

## 2. Pipecat Compatibility Status

### `GeminiLiveLLMService` — Already Available

**Location:** `pipecat.services.google.gemini_live.llm`
**Import:** `from pipecat.services.google.gemini_live import GeminiLiveLLMService`
**Available since:** ~v0.0.50 (originally as `GeminiMultimodalLiveLLMService`, renamed in v0.0.90)

This is a **speech-to-speech (S2S) service** that replaces STT + LLM + TTS in a single component.

### Constructor

```python
GeminiLiveLLMService(
    api_key=os.getenv("GOOGLE_API_KEY"),
    system_instruction="Your persona prompt here",
    settings=GeminiLiveLLMSettings(
        model="models/gemini-2.5-flash-native-audio-preview-12-2025",
        voice="Charon",                        # TTS voice
        modalities=GeminiModalities.AUDIO,     # AUDIO or TEXT
        language="en-US",
        temperature=0.7,
        max_tokens=4096,
    ),
    inference_on_context_initialization=True,
)
```

### Settings Dataclass (`GeminiLiveLLMSettings`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `models/gemini-2.5-flash-native-audio-preview-12-2025` | Model ID |
| `voice` | str | `"Charon"` | Voice name |
| `modalities` | GeminiModalities | AUDIO | Output modality |
| `language` | Language \| str | — | Language code |
| `temperature` | float | — | Sampling temperature |
| `max_tokens` | int | 4096 | Max output tokens |
| `vad` | GeminiVADParams | — | Server-side VAD config |
| `context_window_compression` | dict | — | `{"enabled": True, "trigger_tokens": N}` |
| `thinking` | ThinkingConfig | — | Reasoning budget |
| `enable_affective_dialog` | bool | False | Emotional tone adaptation |
| `proactivity` | ProactivityConfig | — | Proactive response config |

### Audio Handling
- **Input:** Accepts `InputAudioRawFrame`, sends as `audio/pcm;rate={frame.sample_rate}` — flexible input rate
- **Output:** Hardcoded **24 kHz**, outputs `TTSAudioRawFrame(sample_rate=24000, num_channels=1)`
- **Built-in transcription:** Both directions, aggregated into sentences

### Pipeline Architecture Change

**Current pipeline (STT → LLM → TTS):**
```python
pipeline = Pipeline([
    transport.input(),
    stt,                    # DeepgramSTTService / GoogleSTTService
    user_aggregator,        # LLMContextAggregatorPair.user()
    llm,                    # OpenAILLMService / GoogleLLMService
    tts,                    # CartesiaTTSService / GeminiTTSService
    assistant_aggregator,   # LLMContextAggregatorPair.assistant()
    transport.output(),
])
```

**Gemini Live pipeline (single S2S service):**
```python
pipeline = Pipeline([
    transport.input(),
    context_aggregator.user(),
    gemini_live,             # GeminiLiveLLMService (handles STT+LLM+TTS)
    transport.output(),
    context_aggregator.assistant(),
])
```

---

## 3. Audio Format Compatibility

| | Current System | Gemini Live Input | Gemini Live Output |
|-|----------------|-------------------|--------------------|
| **Format** | PCM16 LE | PCM16 LE | PCM16 LE |
| **Sample Rate** | 16 kHz or 24 kHz | 16 kHz | 24 kHz (hardcoded) |
| **Channels** | Mono | Mono | Mono |

**Compatibility assessment:**
- Input audio from MeetingBaas at 16 kHz → directly compatible with Gemini Live input
- Output at 24 kHz → requires `audio_out_sample_rate=24000` on transport (already supported, used for Google TTS)
- **No resampling needed** — the asymmetric 16kHz-in/24kHz-out is handled natively

### VAD Consideration
- Gemini Live has server-side VAD, but pipecat's `GeminiLiveLLMService` does **not** emit `UserStarted/StoppedSpeakingFrame`
- Local Silero VAD (16 kHz) should still be kept for pipeline turn tracking
- Both can coexist: Silero for local frame management, Gemini for server-side interruption

---

## 4. Authentication Requirements

| Requirement | Value |
|-------------|-------|
| **API Key** | Standard `GOOGLE_API_KEY` (same key used for existing Google services) |
| **Key source** | [Google AI Studio](https://aistudio.google.com/apikey) |
| **Pipecat usage** | Pass as `api_key` parameter to `GeminiLiveLLMService` |
| **Additional setup** | None — no service accounts, OAuth, or special permissions needed |
| **Vertex AI** | Alternative: uses OAuth/service account via `GeminiLiveVertexLLMService` |

The existing `GOOGLE_API_KEY` environment variable works as-is. When using Gemini Live as the sole provider, Deepgram and Cartesia API keys become unnecessary.

---

## 5. Recommended Integration Approach

### Phase 1: Add `gemini-live` as a new provider option (Estimated: 1-2 days)

1. **Add import** for `GeminiLiveLLMService` alongside existing Google imports in `scripts/meetingbaas.py`
2. **New provider path:** When `LLM_PROVIDER=gemini-live`:
   - Skip separate STT and TTS initialization
   - Create `GeminiLiveLLMService` with persona's system prompt
   - Build simplified pipeline (transport → context.user → gemini_live → transport.output → context.assistant)
   - Force `audio_out_sample_rate=24000`
3. **Persona config:** Add `gemini_live_voice` field to persona metadata (e.g., "Charon", "Puck")
4. **Environment variables:**
   - `LLM_PROVIDER=gemini-live` (new value)
   - `GOOGLE_API_KEY` (existing)
   - `GOOGLE_LIVE_MODEL` (optional, default: `models/gemini-2.5-flash-native-audio-preview-12-2025`)

### Phase 2: Session management & resilience (Estimated: 1 day)

1. **Session resumption** — handle the ~10 min WebSocket lifetime limit
2. **Context compression** — enable sliding window for long meetings
3. **Graceful reconnection** on `goAway` server messages

### Phase 3: Advanced features (Estimated: 1 day)

1. **Transcription forwarding** — pipe input/output transcriptions to meeting transcript
2. **Function calling** — re-enable tools (weather, time) through Gemini's native function calling
3. **VAD tuning** — optimize Gemini's server-side VAD parameters for meeting context

### Key Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Preview model instability | API changes, deprecations | Pin model version, monitor deprecation notices |
| 15-min session limit | Long meetings cut off | Session resumption + context compression |
| Rate limits on free tier | Development throttling | Use paid tier for production; free tier for dev |
| Function calling issues | Known pipecat issues (#1375, #1606) | Can defer tools to Phase 3 |
| No local VAD events from service | Pipeline turn tracking gaps | Keep Silero VAD alongside |

---

## 6. Cost Comparison

### Current Pipeline (separate services)
- Deepgram STT: ~$0.0043/min
- OpenAI GPT-4 Turbo: ~$0.01-0.03/min (varies by verbosity)
- Cartesia TTS: ~$0.006/min
- **Total: ~$0.02-0.05/min**

### Gemini Live (single service)
- Audio input tokens: ~$3/M tokens
- Audio output tokens: ~$12/M tokens
- Approximate: **~$0.01-0.03/min** (depends on conversation density)
- **Eliminates** need for Deepgram, Cartesia, and OpenAI API keys

---

## 7. Summary

| Aspect | Status |
|--------|--------|
| **API maturity** | Preview (stable enough for production with caveats) |
| **Pipecat support** | Full — `GeminiLiveLLMService` in v0.0.105 |
| **Audio compatibility** | Native — 16kHz in, 24kHz out, both PCM16 |
| **Auth compatibility** | Same `GOOGLE_API_KEY` |
| **Integration effort** | ~3-4 days total (Phase 1 is 1-2 days) |
| **Key advantage** | Lower latency, single API, built-in interruption handling |
| **Key limitation** | ~15 min sessions (mitigated by resumption), preview status |
