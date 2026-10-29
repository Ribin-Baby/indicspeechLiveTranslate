# Indic speech call translator

A real-time speech-to-speech call translator: everyone on a call speaks their own language
and hears the others in the language they choose (tested with two and three people). A call
orchestrator built on LiveKit chains speech recognition, translation and speech synthesis.
See [Translated calls](#translated-calls-livekit) for the call stack.

## Models in use

**Since 2026-10-27, for now, calls run on meetmind's model servers.** `.env` sets
`COMPOSE_FILE=docker-compose.yml:docker-compose.shared.yml`, so this repo runs only the call
stack (agent, page, HTTPS proxy). It uses the models that meetmind, the meeting app on this
host, deploys and uses itself:

| Step | Model | Where it runs | API |
|---|---|---|---|
| Noise removal | NVIDIA Maxine BNR (v2, 48 kHz), on each caller's mic before speech detection; since 2026-10-28 | Maxine BNR NIM on another GPU host (`BNR_GRPC_URL` in `.env`) | gRPC `EnhanceAudio` stream, 10 ms chunks (`services/s2s/app/bnr.py`) |
| Speech recognition | NVIDIA Parakeet 1.1B RNNT multilingual, offline mode | Riva runtime on this host's GPU (meetmind's `deploy/gpu-host`, service `asr`), `127.0.0.1:8001` | `POST /v1/audio/transcriptions`, one request per utterance (the same API as the IndicConformer NIM) |
| Translation | **Qwen** `qwen3.8-27b`, a chat LLM with an interpreter instruction (thinking off). Since 2026-10-28; Riva Translate 1.6B before, and still the default without the settings | vLLM on another GPU host (`S2S_TRANSLATE_API=chat`, `S2S_TRANSLATE_URL`, `S2S_TRANSLATE_MODEL` in `.env`) | OpenAI chat completions, streamed |
| Translation (default) | NVIDIA Riva Translate 1.6B (`megatronnmt_any_any_1b`: 36 languages, any direction) | Riva runtime on this host's GPU (meetmind's service `nmt`), `127.0.0.1:8002` | `POST /v1/text/translations`, the Riva Translate NIM's API |
| Speech | NVIDIA Magpie TTS Multilingual | Magpie TTS NIM on another GPU host (`MAGPIE_TTS_URL` in `.env`) | `POST /v1/audio/synthesize_online`, raw PCM streamed |
| WebRTC | LiveKit | meetmind's LiveKit server, `127.0.0.1:7880` | LiveKit |

Callers speak and hear **English or Hindi**, the languages all these models cover. With Qwen,
the translation starts playing ~1.15–1.35 s after a speaker stops, and the other caller hears
it after ~1.4–1.5 s. With Riva it's ~0.9 s and ~1.1 s, but Qwen's translations read more
naturally and hold up better against recognition slips (see
[Translated calls](#translated-calls-livekit)). The two model servers on this GPU take
~22 GB (Parakeet ~12 GB, Riva Translate ~9.6 GB); the call stack itself needs no GPU.

**Architecture diagram** of this pipeline: [docs/live-pipeline.html](docs/live-pipeline.html)
(open it in a browser). It covers:

- a system map of what runs where;
- the path one sentence takes through the agent, with measured times;
- a to-scale latency timeline;
- the components, per-caller settings, caption messages, failure handling and configuration.

## This repo's own model stack (off for now)

Remove `COMPOSE_FILE` from `.env` to run the calls on the models in this repo instead:
5 languages to speak, the 22 scheduled Indian languages plus English to hear. Their images
were removed on 2026-10-27 to free disk, so the next start pulls and builds them again
(~56 GB). They use the same ports (8001, 50051, 8002) and LiveKit ports as meetmind's
servers, so the two can't run side by side.

The models run in **NVIDIA NIM containers** wherever NVIDIA supports them. The one
piece no NIM covers, turning indic-speak's audio tokens into sound, is our own
container.

**Architecture diagram** of this stack: [docs/architecture.html](docs/architecture.html)
(open it in a browser; also published at https://claude.ai/artifact/PUtZvT23wvBJFamux4WsYE).
It shows every component and what flows between them, and a to-scale timeline of one
sentence from the speaker stopping to the listener hearing the translation.

| Service | Model | Port | Runtime | API |
|---|---|---|---|---|
| `asr` | AI4Bharat IndicConformer, one 120M Conformer-CTC per language ([`ai4bharat/indicconformer_stt_<lang>_hybrid_ctc_rnnt_large`](https://huggingface.co/ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large)): Malayalam, Hindi, Kannada, Telugu, Tamil | 8001 (HTTP), 50051 (gRPC) | NVIDIA Riva ASR NIM, custom models (nemo2riva → riva-build, TensorRT) | `POST /v1/audio/transcriptions`, one request per utterance of ≤16 s |
| `translate` | [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate) (Gemma-4-E4B, 22 Indian languages + English, direct between any two) | 8002 | NVIDIA model-free LLM NIM 2.0.13 (vLLM 0.28) | OpenAI chat completions, SSE: text deltas out |
| `tts-lm` | [`bodhan-ai/indic-speak`](https://huggingface.co/bodhan-ai/indic-speak)'s speech LM (Llama-3.2-3B emitting audio tokens) | 8004 | NVIDIA model-free LLM NIM 2.0.13 | OpenAI completions, token ids in and out |
| `tts` | indic-speak's audio decoder (SNAC + Vocos) and the TTS API (22 Indian languages + English) | 8003 | our image: Python + CUDA torch 2.13 | WebSocket / chunked HTTP: PCM out |

All four share one NVIDIA L40S (46 GB) and use 40.3 GB together. On disk, the
Riva NIM image takes 37 GB (it is the Whisper Large v3 NIM image, used only as the Riva
runtime) and the model-free NIM image 43.8 GB (one copy, used by both `translate` and
`tts-lm`).

History, all on 2026-10-24: the first setup ran IndicConformer in our own ONNX Runtime
container and Bodhan AI indic-translate (code in
`archive/previous-setup-2026-10-24.tar.gz`; [docs/NIM_POC.md](docs/NIM_POC.md) showed the
model-free NIM serves a local checkpoint unchanged). Then ASR moved to NVIDIA's Whisper
Large v3 NIM, which wrote Malayalam in Gurmukhi script (~92% character errors), so
IndicConformer came back, this time inside NVIDIA's Riva NIM. Translation used
sarvamai/sarvam-translate until 2026-10-26. In a real Malayalam ↔ Kannada call it invented
meanings and sometimes commented on the text instead of translating it, so indic-translate
came back, now on the NIM with the same deployment and parallel translation. On 2026-10-27
the calls moved to meetmind's models (above), English and Hindi only, for now.

```
indicspeechModels/
├── docker-compose.yml        # model services + call stack, started in order
├── docker-compose.shared.yml # for now: only the call stack, on meetmind's model servers
├── models/                   # weights (mounted read-only at /models): indic-translate, indic-speak, snac_24khz, indicconformer-<lang>
├── nim-cache/                # the LLM NIMs' caches (compile caches, runtime manifests)
├── nim-export/asr-indic/     # the ASR NIM's model repository (custom_model.tar.gz, built)
├── services/tts/             # indic-speak audio decoder + TTS API (Dockerfile + app/)
├── services/asr-nim/         # ASR NIM healthcheck (warm-up, self-restart) + build/ (IndicConformer → Riva)
├── services/translate-nim/   # healthcheck + warm-up for the translate NIM
├── services/s2s/             # call orchestrator: LiveKit agent worker (incl. VAD) + demo web app
├── services/{livekit,proxy}/ # LiveKit server config, HTTPS proxy (Caddy) config
├── clients/                  # clients, benchmarks, call tests
├── archive/                  # the previous setup's code
├── build/asr-nim/            # scratch space of the ASR build
└── testdata/                 # sample clips + outputs
```

## Quick start

### On meetmind's model servers (current)

This needs, on this host:

- **meetmind's model servers running:** in `meetmind/deploy/gpu-host`, run `docker compose --env-file .env up -d`, which starts `asr` and `nmt`.
- **meetmind's LiveKit accepting this repo's key:** add `LIVEKIT_API_KEY: LIVEKIT_API_SECRET` under `keys:` in meetmind's `infra/livekit.yaml`, then restart `meetmind-livekit`.
- **These settings in `.env`:**
  - `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and `PUBLIC_HOST`;
  - `MAGPIE_TTS_URL`, the Magpie NIM (`http://host:9000/v1`);
  - `COMPOSE_FILE=docker-compose.yml:docker-compose.shared.yml`.

```bash
docker compose up -d --build
```

This starts `s2s-agent`, `s2s-web` and `proxy`; the call page is `https://$PUBLIC_HOST:8443`.
An English ↔ Hindi call test:

```bash
docker run --rm --network host -v $PWD/clients:/clients -v $PWD/testdata:/td indic-s2s:0.1 python /clients/call_test_en_hi.py
```

### With this repo's own models

Remove `COMPOSE_FILE` from `.env`, and stop meetmind's model servers and LiveKit first (same
ports). The NIM images come from NVIDIA's registry. Log in once, with the NGC API key
(`NVIDIA_API_KEY` in `.env`) as the password:

```bash
docker login nvcr.io -u '$oauthtoken'
```

The LLM NIMs keep their caches in `nim-cache/` (NVIDIA's `LOCAL_NIM_CACHE` pattern). The
folders must be writable by uid 1000, the user the NIM containers run as:

```bash
mkdir -p nim-cache/translate nim-cache/tts-lm
```

The ASR NIM loads a model repository built from AI4Bharat's checkpoints. Each one is gated
on Hugging Face: accept its terms (for example
https://huggingface.co/ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large) with the
account whose `HF_TOKEN` is in `.env`, then build (about 3 minutes per language, on the GPU):

```bash
services/asr-nim/build/build.sh ml hi kn te ta
```

```bash
docker compose up -d --build
```

```bash
docker compose ps
```

The first start takes about 4 minutes, mostly the LLM NIMs compiling CUDA graphs; the
ASR NIM is ready in ~35 s. Services start one after another and report `healthy` only after their
warm-up: the ASR and translate healthchecks send warm-up requests themselves
([services/asr-nim/healthcheck.py](services/asr-nim/healthcheck.py),
[services/translate-nim/healthcheck.py](services/translate-nim/healthcheck.py)), and the
`tts` service warms `tts-lm` up when it starts.

Client tools run from the project venv (`.venv`):

```bash
.venv/bin/python clients/asr_client.py testdata/sample_hindi.wav --language hi-IN
```

```bash
.venv/bin/python clients/translate_client.py --text "नमस्ते, आप कैसे हैं?" --tgt en
```

```bash
.venv/bin/python clients/tts_stream.py --text "வணக்கம்" --language ta --out ta.wav
```

The model services bind to `127.0.0.1` because their APIs have no
authentication. Set `BIND_ADDR=0.0.0.0` only behind something that adds auth.
The call stack reaches them on localhost. The only ports on the network are the
call demo's: `8443` (HTTPS page and LiveKit signaling, token-gated), `7881/tcp`
and `50000-50200/udp` (WebRTC media).

Language codes: ASR takes Riva codes (`ml-IN`, `hi-IN`, …), TTS short ISO codes (`hi`,
`ta`, `en`, …), translate language names (`Hindi`, `Tamil`, `English`, …).

---

## Own model stack: APIs and measurements

These sections describe this repo's own models (off for now, see above). With the shared
setup, ports 8001 and 8002 are meetmind's Parakeet and Riva Translate servers, whose APIs are
in the [Models in use](#models-in-use) table.

> **Using the models directly:** [docs/MODEL_SERVICES.md](docs/MODEL_SERVICES.md) is the
> complete guide to the own stack's model APIs (every parameter, error and tip), with tested
> examples in [docs/examples/](docs/examples/). The sections below are a summary.

### ASR — `http://host:8001/v1/audio/transcriptions`

AI4Bharat IndicConformer models served by NVIDIA's Riva ASR NIM, one model per language,
deployed the way NVIDIA documents custom models (see
[Implementation notes](#implementation-notes)). Each request transcribes a finished piece
of audio; the call agent cuts live audio into utterances with a VAD first (see
[Per utterance](#translated-calls-livekit)).

```bash
curl -s localhost:8001/v1/audio/transcriptions -F language=hi-IN -F file=@testdata/sample_hindi.wav
```

```json
{"text":"नमस्ते आज हम गणित की कक्षा सुरू करेंगे कृपया अपनी पुस्तके खोलिए और पहला पाठ पढ़िय "}
```

- `file`: 16-bit PCM WAV, mono, 16 kHz. **At most 16 s per request**: the models read
  the whole utterance in one pass, and a longer request crashes Riva's inference server.
  The container then restarts itself (back in ~55 s), but every caller loses speech
  recognition meanwhile. `docs/examples/asr_transcribe.py` cuts longer files at pauses.
- `language`: `ml-IN`, `hi-IN`, `kn-IN`, `te-IN` or `ta-IN`. Plain `hi` gives
  `404 Model not found for language hi`. There is no English and no auto-detection.
- More languages: accept the checkpoint's terms on Hugging Face, add the language to
  `services/asr-nim/build/build.sh`'s arguments, rebuild, restart `asr`.
- Also: gRPC on `50051` (the Riva ASR API, e.g. `nvidia-riva-client`; not tested here),
  `GET /v1/health/ready`.

### Translate — `http://host:8002/v1/chat/completions`

bodhan-ai/indic-translate on NVIDIA's model-free LLM NIM, an OpenAI-compatible server, so
any OpenAI client works. The caller builds the prompt, following the model's prompt
contract (`models/indic-translate/scripts/prompt_contract.py`):

```json
{"model": "indic-translate", "stream": true, "temperature": 0, "stop": ["<turn|>"], "max_tokens": 256,
 "messages": [{"role": "user", "content": "Translate the following text into Kannada:\n\nലവണി എന്തെങ്കിലും പറയാമോ."}]}
```

- One `user` message and no `system` message. It names only the **target** language;
  the model infers the source. It translates **directly between any two** of the 22
  Indian languages and English.
- Target names are the model's: `Tamil`, `Hindi`, `English`, and
  `Kashmiri (Perso-Arabic script)`, `Manipuri (Meitei script)`,
  `Sindhi (Devanagari script)` for the languages written in two scripts.
  `Hindi (Roman)` and `Hindi (code-mixed)` also work. The agent's ISO-code → name table
  is `PROMPT_NAME` in `services/s2s/app/languages.py`.
- **End every sentence with punctuation.** Speech recognition writes none, and without
  it the model often swaps names (see "Cleaning up speech-recognition text" below).
- `stream: true` returns Server-Sent Events whose `choices[0].delta.content` pieces
  make up the translation, ending with `data: [DONE]`.
- Concurrent requests share one batch on the GPU; there is no batch endpoint.

Also: `GET /v1/models`, `GET /v1/health/ready`, `GET /v1/metrics` (Prometheus).

### TTS — `ws://host:8003/v1/stream`

| client → server | |
|---|---|
| `{"type": "synthesize", "id": "u1", "text": "...", "language": "ta", "gender": "female", "sample_rate": 16000}` | queued; one at a time per socket, in order |
| `{"type": "cancel"}` | barge-in: stop the current request and drop the queue |
| `{"type": "cancel", "id": "u1"}` | cancel one request |

The server answers `{"type": "queued", "id", "position"}`, then
`{"type": "start", "id", "sample_rate", "encoding", "speaker"}`, then **binary
frames of PCM s16le**. A frame always belongs to the most recent `start`. It ends
with `{"type": "done", "id", "audio_ms", "ttfa_ms", "latency_ms", "rtf", "tokens"}`
(or `cancelled` / `error`).

Optional fields: `speaker` (any name from `GET /v1/voices`; the default is the
language's recommended female/male voice), `style` (`"news"`, `"Customer Care"`,
`"happy"`, …; preview quality per the model card), `sample_rate` (24000 native,
16000 / 8000 resampled), `format` (`pcm_s16le` | `pcm_f32le`), `temperature`,
`top_p`, `top_k`, `repetition_penalty`, `seed`, `max_seconds`.

The service splits long text into sentence chunks of ≤300 characters and streams
them back to back.

HTTP: `POST /v1/tts` with the same fields plus `format` (`wav` | `pcm_s16le` |
`pcm_f32le`) and `stream`. Non-streaming responses carry `X-TTFA-Ms`,
`X-Latency-Ms`, `X-Audio-Seconds` headers.

**How it runs:** the speech LM is in the `tts-lm` NIM. The `tts` service sends it the
prompt as token ids on `/v1/completions` (`return_token_ids: true`, the model card's
sampling: temperature 0.6, top-p 0.9, repetition penalty 1.2) and streams the
generated audio-token ids back. It turns them into sound itself (SNAC codes →
Vocos), on the GPU, in streamed windows.

---

### Measured performance

These are measurements on one L40S running all the services at once. Clients ran
on the same host and were driven in real time.

**ASR.** Each request is the same 6.5 s Hindi clip.

| concurrent requests | latency p50 | max | throughput |
|---|---|---|---|
| 1 | 25 ms | — | — |
| 4 | 61 ms | 95 ms | 42 clips/s |
| 8 | 105 ms | 183 ms | 44 clips/s |
| 16 | 194 ms | 367 ms | 44 clips/s |

Character error rate on our TTS test clips (3–5 sentences per language; TTS speech is
cleaner than a real caller, so treat these as a floor, not a forecast):

| | Malayalam | Hindi | Kannada | Telugu | Tamil |
|---|---|---|---|---|---|
| IndicConformer on Riva | 0.0% | 0.0% | 0.0% | 1.0% | 0.7% |
| Whisper Large v3 NIM (before) | 91.7% | 16.5% | 10.7% | 17.3% | 8.1% |

The Riva results are identical to the same models run in NeMo.

**Translate** (indic-translate). Sentences are ~10–15 words, Indian language → Indian
language directly, each request its own streamed HTTP call.

| concurrency | first text p50 | full sentence p50 | throughput |
|---|---|---|---|
| 1 | 45 ms | 373 ms | — |
| 8 | 73 ms | 260 ms | 20 sentences/s |
| 32 | 114 ms | 322 ms | 68 sentences/s |
| 64 | 187 ms | 429 ms | 104 sentences/s |

**Translation quality on a real call.** We replayed the 38 recognized utterances of a
live Malayalam ↔ Kannada call through both translators (36 had text that could be
judged):

| | right | partly right | wrong | invented content or commentary |
|---|---|---|---|---|
| sarvam-translate (via English) | 15 | 8 | 13 | 6 |
| indic-translate (direct) | 15 | 14 | 7 | 0 |

indic-translate stays faithful. When the recognized text is garbled, or English words
were written phonetically in Kannada or Malayalam script, it copies them over
phonetically instead of translating. Most of the remaining errors start in speech
recognition, which garbles casual speech and English words mixed into it.

**TTS.** Each stream is one sentence of 7–9 s of audio. The real-time factor
(RTF) is generation time divided by audio length, so above 1.0 the listener
eventually hears gaps.

| concurrent streams | first audio p50 | RTF | playback stalls |
|---|---|---|---|
| 1 | 376 ms | 0.84 | none |
| 4 | 418 ms | 0.96 | none |
| 8 | 422 ms | 1.05 | up to 425 ms |

These match the earlier in-process vLLM setup (380 ms, 0.84 for one stream).

**Capacity.** TTS is the limit: about 6–7 simultaneous real-time voices. Because callers
take turns speaking, that covers more calls than voices. Speech recognition handles ~44
utterances a second, so it is no longer a constraint.

Re-run the benchmarks:

```bash
.venv/bin/python clients/asr_client.py testdata/sample_hindi.wav --language hi-IN --bench --concurrency 1 4 8 16
```

```bash
.venv/bin/python clients/translate_client.py --bench --concurrency 1 8 32 64
```

```bash
.venv/bin/python clients/tts_stream.py --bench --concurrency 1 4 8
```

## Translated calls (LiveKit)

```
browser A ──mic──┐                                   ┌──► "translation-A" ──► browser A
                 ▼                                   │     (only A may subscribe)
          LiveKit server (self-hosted SFU) ◄────► s2s-agent ──► ASR → translate → TTS
                 ▲                                   │     (the models in use, above)
browser B ──mic──┘                                   └──► "translation-B" ──► browser B
```

| Service | What it is | Listens on |
|---|---|---|
| LiveKit | now meetmind's server (WebRTC media server), shared; this repo's own `livekit` service (`livekit/livekit-server:v1.13.9`) runs only with its own model stack | signaling `127.0.0.1:7880`; media UDP `50000-50200`, TCP `7881` on `PUBLIC_HOST` |
| `s2s-agent` | LiveKit agent worker (`services/s2s/app/agent.py`); joins the rooms that ask for it | outbound only |
| `s2s-web` | token API + demo page (`services/s2s/app/web.py`, `static/`) | `127.0.0.1:8090` |
| `proxy` | Caddy, HTTPS with its own CA, in front of the page and `/rtc` | `8443` |

These services run with host networking (WebRTC needs real addresses) and
need no GPU. `.env` holds `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` and
`PUBLIC_HOST` (the address browsers use, now `10.79.252.124`).

**Which models the agent uses** is set in its environment (`docker-compose.shared.yml` sets
the current ones):

| Setting | Current (shared) | Own model stack |
|---|---|---|
| `ASR_URL` | Parakeet, `http://127.0.0.1:8001` | IndicConformer NIM, same address and API |
| `TRANSLATE_API`, `TRANSLATE_URL`, `TRANSLATE_MODEL` | `chat`, Qwen's `/v1` URL, `qwen3.8-27b` (from `S2S_TRANSLATE_API`, `S2S_TRANSLATE_URL`, `S2S_TRANSLATE_MODEL` in `.env`); without those, `riva`, `http://127.0.0.1:8002` | `openai`, `http://127.0.0.1:8002/v1`, `indic-translate` |
| `TTS_API` | `magpie` (`MAGPIE_TTS_URL`; each caller picks a voice, and `MAGPIE_FEMALE_VOICE` / `MAGPIE_MALE_VOICE` set the defaults, Louise and Jason) | `indic-speak` (`TTS_URL`) |
| `S2S_SPEAK_LANGUAGES`, `S2S_HEAR_LANGUAGES` | `en,hi` | unset: every language the models cover |
| `BNR_GRPC_URL` (`BNR_INTENSITY`, default 1.0) | the Maxine BNR NIM (`host:8001`): noise removal on | the same; unset: off |
| `S2S_VAD_MIN_SILENCE_MS`, `S2S_PHRASE_S`, `S2S_BREATH_MS` | 350 ms pause ends an utterance; after 3 s of speech, a 200 ms breath does | the same |

**Sharing a LiveKit server.** The agent registers as `indic-s2s` (`S2S_AGENT_NAME`) and joins
only rooms that ask for it by that name: the web app puts the request in every caller's
token. Its watchdog re-dispatches only those rooms. On meetmind's LiveKit server, meetmind's
calls are left alone, and meetmind ignores this app's rooms.

**Try it:** open `https://10.79.252.124:8443` on two devices, or two browsers
with different mics. Pick a name, your language and the language to hear the
other person in, check your mic, and join; the **Invite** button copies the room
link for the other person. In the call, each person's ring pulses with their
voice, and each caption shows its state (listening, translating, playing) and
its delay. **Settings** changes languages, your voice (with a **Play** sample), background
noise removal for your mic, mic and speaker devices, volumes, and whether captions show the
English step and the original words. The certificate comes
from Caddy's local CA, so browsers warn once. To remove the warning, trust its
root certificate:

```bash
docker compose cp proxy:/data/caddy/pki/authorities/local/root.crt ./caddy-root.crt
```

On a machine that can't reach `10.79.252.124` directly, testing needs that
address routable. Reaching the call from the internet needs a TURN server and a
real certificate; neither is set up yet.

**Language, voice and noise removal.** Each caller has four participant attributes. They're
set at join time and can be changed mid-call from the page's settings.

- `speak`: now English or Hindi. With the own models, the languages with an ASR model:
  Malayalam, Hindi, Kannada, Telugu, Tamil.
- `hear`: now English or Hindi. With the own models, the 22 Indian languages plus English.
- `voice`: the voice the *other* people hear them in (`languages.VOICES`).
  - With Magpie, it's one of the speakers that exist in every language the call can be heard
    in: Louise, Aria, Sofia, Isabela or Leo (female), Jason, Pascal or Diego (male).
  - NVIDIA doesn't publish their genders; these groups come from each voice's measured pitch.
    Leo, despite the name, measures 216–235 Hz, so the default male voice is Jason.
  - The page's **Play** button fetches a short sample (`GET /api/voice-preview`).
  - With the own models, the choice is `female` or `male` (indic-speak's recommended voices).
- `bnr`: `on` or `off`, background noise removal for this caller's mic. On by default; offered
  only when `BNR_GRPC_URL` is set.

The agent restarts that caller's speech recognition (new language or noise removal) or
retargets translation and voice on the fly. If a
listener's `hear` equals the speaker's `speak`, nothing is translated and the page
plays the speaker's own voice. Otherwise the page plays only the translation; an
"original voice" slider can mix the speaker's own voice back in.

The next two paragraphs are about indic-translate (own model stack). Riva Translate
translates every pair in one call, takes the text as it is, and returns the whole translation
at once (`RivaTranslateClient` in `agent.py`).

**Translation route.** `S2S_PIVOT_THROUGH_ENGLISH` in the agent's environment picks the
route between two Indian languages:

```bash
S2S_PIVOT_THROUGH_ENGLISH=0   # direct, source -> target (default)
S2S_PIVOT_THROUGH_ENGLISH=1   # source -> English -> target, as with sarvam-translate
```

Direct is one model call and keeps meaning and register; going through English can
lose register (a formal "you" came back as the informal "உனக்கு" in Tamil). With the
pivot on, the English step runs once per sentence and serves every listener language,
and captions can show it ("via English"). Translations into or out of English are always
one call. Set it in `.env` and run `docker compose up -d s2s-agent`.

**Cleaning up speech-recognition text for the translator** (`TranslateClient` in
`agent.py`):

- *End mark.* ASR writes no punctuation, and without a sentence end indic-translate
  often swaps names and places. In a live call, "…നാളെ പാലക്കാട് പോയാലോ…" ("…go to
  Palakkad tomorrow…") came out as "let's all think about going to new york tomorrow";
  other tests gave Mumbai, Chennai, Shimla, "John", even "Trump". So the agent adds "।"
  (Hindi, Bengali, Odia, Punjabi…) or "." first. In our tests that fixed every swapped
  name (0 of 63 into English, against most of them before).
- *Respelling.* Between two Indian languages, ~5% of outputs (more on garbled speech)
  keep a word, often a name, in the source script. That reads wrong and makes TTS babble,
  so the agent respells those letters in the target script (Unicode keeps the Indian
  scripts letter for letter at the same offsets): "മൈസൂರಿಗೆ" → "മൈസൂരിഗെ". Over 252 test
  translations, none were left in mixed scripts. Retrying without the end mark was worse,
  because it sometimes brought the name swap back.

**Background noise removal** (`services/s2s/app/bnr.py`, when `BNR_GRPC_URL` is set). Each
mic, after echo cancellation, streams through NVIDIA's Maxine BNR NIM in 10 ms chunks. That's
16 kHz up to the model's 48 kHz and back, with LiveKit's resampler at its no-delay setting.
Echo cancellation comes first because it models the echo as a copy of the playout, which
denoised audio no longer is. Measured on 2026-10-28:

- Each 10 ms chunk takes ~13 ms round trip, and the cleaned audio lags by ~40 ms (the model's
  lookahead). The noise floor drops ~37 dB while speech keeps its level.
- Final transcripts arrive ~70 ms later: a median 718 ms after the speaker stops, against
  649 ms without it (`clients/asr_timing_test.py`, 8 utterances each). In full calls on clean
  clips, the other caller heard the translation after 1.45–1.58 s with it and 1.37–1.49 s
  without it, ~60–90 ms more (`clients/call_test_en_hi.py`, two runs each way).
- Hindi with fan noise, hum and keyboard clicks mixed in at 5 dB SNR
  (`testdata/sample_hindi_noisy.wav`, `--hindi /td/sample_hindi_noisy.wav`):
  - With it, the meaning came through in all 4 runs ("…गणित की कक्षा शुरू करेंगे…" → "…begin
    our math class…"). One word slipped twice ("कृतिया" for "कृपया") and was still translated
    as "please".
  - Without it, 4 of 6 runs had wrong words ("दनिक/दमित की कक्षा", "फिर क्या" for "कृपया"),
    and one became "the class on excrement" ("गन्द की कक्षा").
  - The English noisy clip was recognized correctly either way.

If the NIM is unreachable, errors, or leaves 0.5 s unanswered, that mic goes back to raw audio,
so a call never stops because of noise removal. The latency figures still count from when the
caller actually stopped: the agent maps the cleaned audio back to the mic's own clock, minus the
model's lag.

**Per utterance:** the agent takes the caller's mic (16 kHz; echo-cancelled when playout
leaks into it, see below; noise removed, see above), runs Silero VAD on it (`services/s2s/app/vad.py`, CPU) and
sends each finished utterance to the ASR NIM (`services/s2s/app/asr.py`). On each final
transcript, it translates into every listener language at the same time (streaming the
text as captions), and each language goes to TTS as soon as it is ready: **once per
language**, with the audio copied to every listener of that language (`Synthesizer`, or
`MagpieSynthesizer` for Magpie, in `agent.py`). Each listener has their own track, which only they may subscribe to
(publisher-side permissions), and their own playback queue that plays utterances in
order. So one person speaking means one voice per distinct listener language, at most
N−1 in a call of N people. With the own models, TTS (6–7 real-time voices) keeps calls with up to about 5–7
listener languages smooth, and people sharing a language cost nothing extra. A call never
runs more syntheses at once than it has listeners.
Transcripts, translations and per-utterance latency go to every caller on the
`captions` data topic: `transcript`, `translation`, `metrics` and `status`
messages, see `agent.py`.

Live captions of the speaker's own words while they talk (`S2S_ASR_PARTIALS=1`, on):
the utterance so far is re-transcribed every half second (~50 ms per request on Parakeet,
~25 ms on IndicConformer).

**Measured now** (shared models) with two simulated callers in real time
(`clients/call_test_en_hi.py`, three runs on 2026-10-27). Times are from the moment the
speaker stops talking:

| | Hindi → English (6.5 s clip) | English → Hindi (~5 s) |
|---|---|---|
| final transcript | 0.56–0.61 s (the 0.5 s VAD pause; ASR adds ~50 ms) | 0.54–0.56 s |
| translation complete | 0.83–0.88 s | 0.83–0.93 s |
| translated speech starts (agent) | 0.87–0.92 s | 0.89–0.99 s |
| translated speech heard (other caller, after WebRTC) | 1.12–1.14 s | 1.04 s |

Most of what is left is the pause that ends an utterance (`S2S_VAD_MIN_SILENCE_MS`, 0.5 s
when this was measured, 0.35 s now, see below) and translation (~0.3 s); Magpie's first audio
takes ~50 ms.

**Qwen as the translator** (`S2S_TRANSLATE_API=chat`, qwen3.8-27b, three runs each, back to
back with Riva on 2026-10-27): heard after 1.36–1.41 s (Hindi → English) and 1.50–1.53 s
(English → Hindi), against Riva's 1.14–1.20 s and 1.09–1.13 s. The agent speaks a
translation only once it is complete, and Qwen takes ~0.5–0.7 s for these sentences
against Riva's ~0.3 s. Its translations read more naturally and held up better against
recognition slips:

| | Qwen | Riva |
|---|---|---|
| Hindi clip ("…पहला पाठ पढ़िए", recognized with slips) | "…Please open your books and read the first page." | "…read the first letter" / "…increase the first poem" |
| "Can we move the launch review to Friday at three…?" | "क्या हम लॉन्च रिव्यू को शुक्रवार को दोपहर तीन बजे पर शिफ्ट कर सकते हैं?" | "क्या हम दोपहर तीन बजे शुक्रवार को लॉन्च समीक्षा स्थानांतरित कर सकते हैं?" (formal) |

**Shorter pauses, trimmed voice and catch-up** (2026-10-28, with Qwen and noise removal on):

- *Shorter pause.* A 350 ms pause ends an utterance, down from 500 ms
  (`S2S_VAD_MIN_SILENCE_MS`).
- *Breath cut.* Once someone has talked for 3 s, a 200 ms breath ends it too (`S2S_PHRASE_S`,
  `S2S_BREATH_MS`; `vad.SegmenterConfig.phrase_s`). Before, a monologue whose sentences
  were 0.35 s apart was only cut at the 15 s limit, and each 15 s block became ~17 s of Hindi.
- *Trimmed voice.* Magpie's silence before and after each clip is trimmed: speech starts
  sooner, and clips played back to back no longer add 0.3–0.4 s of dead air
  (`speech.SilenceTrimmer`).
- *Catch-up.* A translation that had to wait for the previous one plays faster without changing
  the pitch: 1.06× from a 0.3 s wait, 1.2× at 1 s, up to 1.4× from 2 s (`speech.time_stretch`,
  as in meetmind). The `catch_up` field in `metrics` shows the rate used.

| | Before | After |
|---|---|---|
| Hindi → English call, heard after (2 runs) | 1.45–1.47 s | the first sentence plays while the second is still being spoken; the last one starts 0.68–0.71 s after the speaker stops |
| English → Hindi call, heard after (2 runs) | 1.52–1.55 s | **1.22 s** |
| Final transcript after the speaker stops | 0.63–0.70 s | 0.33 s in long sentences, ~0.46 s in short ones |
| Long English turn heard in Hindi (`clients/long_turn_test.py`, 35 s, 11 sentences): pieces it was cut into | 3 | 9, all at sentence ends |
| ... worst delay of a piece | 17.4 s | 4.7 s |
| ... translated speech still playing after the speaker stopped | 22.8 s | 4.8 s |

**With the own model stack** (`clients/call_test.py`, 2026-10-26):

| | Hindi → Tamil (6.5 s, direct) | Tamil → Hindi (direct) | Hindi → English |
|---|---|---|---|
| final transcript | 0.50 s (the VAD pause; ASR adds ~20 ms) | 0.50 s | 0.52 s |
| translation complete | 0.96 s | 0.67–0.71 s | 0.83 s |
| translated speech starts (agent) | 1.35 s | 1.05–1.10 s | 1.21 s |
| translated speech heard (other caller, after WebRTC) | 1.59 s | — | 1.39 s |

With sarvam-translate (through English) the same turns were heard at 1.83 s and 1.33 s,
and with the Whisper NIM at 2.16 s and 1.79 s. Most of what is left is the 0.5 s pause
that ends an utterance (`S2S_VAD_MIN_SILENCE_MS`), translation (~0.4 s), and TTS until
the first audio (~0.4 s). The browser test (`clients/browser_test.js`, two headless Chromium callers with
WAV files as microphones) showed the same captions and 1.1–1.6 s delays on the real page.

```bash
docker run --rm --network host --env-file .env -v $PWD/clients:/clients -v $PWD/testdata:/td indic-s2s:0.1 python /clients/call_test.py
```

**Three people** (`clients/call_test3.py`): Asha (Hindi), Bala (Tamil) and Chitra
(Malayalam; the clip is our own TTS) each take a turn, each hearing their own language;
then Bala switches to English. Times are when each listener heard the translation, from
the moment the speaker stopped:

| Speaker | First listener | Second listener |
|---|---|---|
| Asha (Hindi) | Bala (Tamil) 1.72 s | Chitra (Malayalam) 1.73 s |
| Chitra (Malayalam) | Asha (Hindi) 1.55 s | Bala (Tamil) 1.68 s |
| Asha (Hindi) | Bala (English) 1.68 s | Chitra (Malayalam) 1.88 s |

All listener languages are translated in parallel (with sarvam-translate through English:
1.77–2.25 s). Before that, the second listener waited for the first: 3.28 s and 3.12 s
in the same turns. Every listener receives only
their own translated track.

```bash
docker run --rm --network host --env-file .env -v $PWD/clients:/clients -v $PWD/testdata:/td indic-s2s:0.1 python /clients/call_test3.py
```

**Two listeners, one language** (`clients/call_test_shared.py`): Asha speaks Hindi while
Bala and Chitra both hear Tamil. One synthesis serves both (identical TTS stats: 9.05 s
of audio, first audio after 371 ms), and both heard it at 1.57 s.

```bash
docker run --rm --network host --env-file .env -v $PWD/clients:/clients -v $PWD/testdata:/td indic-s2s:0.1 python /clients/call_test_shared.py
```

**Echo protection.** A browser's echo cancellation removes only its own speaker's sound.
When two callers sit in the same room, or one plays the call loudly enough to beat
the browser, translated speech leaks into a microphone, gets transcribed and translated
again, and bounces between the two sides indefinitely. The worker prevents this in
two layers:

- It cancels everything it plays into the room out of every microphone before
  speech recognition (WebRTC AEC3 through LiveKit's `AudioProcessingModule`, using a
  playout timeline as the reference). A coarse delay estimator extends the canceller
  from about 0.4 s to 1.5 s of echo path. AEC3 always runs, but speech recognition gets
  its output only while playout is actually leaking into that mic (a strong correlation
  in the last 20 s); otherwise it gets the mic untouched. AEC3 turns down the start of
  speech that overlaps playback even when nothing leaks, and that cost IndicConformer
  the first word of such utterances ("வணக்கம்" in the call test).
- Each finished utterance gets an echo score: how closely its loudness follows the
  playout. Echo scores 0.79–0.99, people talking over the playout 0.39–0.66. An
  utterance at 0.72 or above is not translated, and neither is one above 0.5 that
  repeats words the room just played. The page shows it dimmed as "echo, not translated".

`clients/echo_test.py` reproduces the loop with two simulated callers whose microphones
hear the room. With protection off, one sentence turned into 7 bouncing utterances. With
it on, there was no loop for echo paths of 0.2–1.0 s at up to equal loudness, and real
speech over the playout was still translated. Rerun on the current setup: no loop; the
agent saw the leak within seconds (correlation 0.95–0.98), and the two fragments that got
through before that were dropped as echo. `S2S_ECHO_CANCEL=0` turns the
protection off. `S2S_LOG_TEXT=1` (on) writes each transcript, its echo score and each
translation to `docker compose logs s2s-agent`; turn it off where call content must not
be logged.

**Restarts and crashes.** LiveKit dispatches the translator only when a room is
created, so a worker restart or crash mid-call would leave the room without one.
The web app's watchdog checks every 5 s and dispatches a new translator to any
room that has callers but no translator for 8 s. The worker also leaves calls
within 5 s of being told to stop (`S2S_DRAIN_TIMEOUT_S`). Measured on a restart
during a call: about 10-15 s without translation, then exactly one translator back.

**Not done yet:** phone calls (LiveKit SIP + an Indian SIP trunk), TURN/TLS for
internet use, and a barge-in policy. Today a translation keeps playing if the
listener starts talking.

## Implementation notes

These are the non-obvious decisions. Each one was measured, not assumed. Most are about the
own model stack. The shared setup's model servers are documented in meetmind's
`docs/DEPLOYMENT.md`.

**NVIDIA deployment.** Every model runs in an NVIDIA NIM container, configured the way
NVIDIA documents it: custom ASR models through `nemo2riva` → `riva-build` →
`riva-deploy` and the NIM's `NIM_DISABLE_MODEL_DOWNLOAD` / `NIM_EXPORT_PATH`, local
checkpoints through the model-free LLM NIM's `NIM_MODEL_PATH`, a host cache folder per LLM
NIM at `/opt/nim/.cache`, the standard ports and health endpoints (`/v1/health/ready`).
Two things NIMs don't do are ours: the VAD that turns a live call into utterances (in the
agent), and the SNAC + Vocos audio decoder for indic-speak (the `tts` service).
indic-speak has no Riva or speech-NIM path: Riva TTS only runs NVIDIA's own architectures.

**IndicConformer into Riva (`services/asr-nim/build/`).** Each step was needed:

- *One language per model.* AI4Bharat's checkpoints are hybrid RNNT/CTC models whose
  CTC head covers all 22 languages' tokens (22 × 256 + blank, "multi-softmax"), which
  stock NeMo can't load and Riva can't serve. `make_ctc.py` keeps the encoder and that
  language's 256 rows plus the blank, and saves a standard `EncDecCTCModelBPE`. The
  output is identical to the original model's CTC branch. (Each language's checkpoint is
  its own fine-tune: the Malayalam one's Hindi rows output nothing.)
- *`nemo2riva --max-dim 3200`.* Its default trace length overruns the encoder's position
  table.
- *fp32 TensorRT* (`--nn.use_trt_fp32`). The fp16 engine output only blanks, even with
  NeMo's own features fed straight into it.
- *Whole-utterance offline mode with NeMo's features* (`--streaming=False`, one 16 s
  chunk without padding, `--feature_extractor_type=torch --torch_feature_type=nemo`).
  Riva's default chunked mode with its default features dropped the first word in
  Tamil, Telugu and Kannada. The cost is the 16 s limit per request. riva-build doesn't
  package the mel filterbank file this featurizer reads, so the build makes it.
- *Batch 4.* At batch 16 each fp32 engine reserved ~2.1 GB of working memory, and five
  didn't fit next to the LLMs.

**Speech recognition (`services/s2s/app/asr.py`).** Each caller's stream emits the same
events a streaming ASR would (`speech_start`, `partial`, `final` with stream
timestamps), so echo scoring and latency tracking are unchanged. Finals go out in
utterance order. Utterances are cut at 15 s in a monologue so finals keep coming, and
so no request reaches the 16 s limit. `import silero_vad` pulls in torch, so the agent
loads the package's ONNX file directly and runs it with CPU ONNX Runtime. The first
request after the ASR NIM starts is slow, so its healthcheck warms every language's
model before reporting healthy.

**Riva streaming.** Riva can also stream (partial results while someone talks, its own
end-of-speech detection), but it wouldn't shorten the delay here: whole-utterance
recognition takes ~25 ms, and a final transcript has to wait for the pause either way.
Live captions come from re-transcribing the utterance so far.

**Translate.** The NIM's vLLM 0.28 reserves very little KV cache at a GPU share
(~1.5 GiB at 0.50 for indic-translate), so the size is pinned with
`--kv-cache-memory-bytes` (3 GiB, 24,111 tokens; small enough that the model fits next
to ASR and TTS). `--language-model-only` skips Gemma-4's vision and audio encoders. The
first burst of concurrent requests stalls ~1 s while batched-decode kernels compile,
which is why the healthcheck sends one before reporting healthy.

**TTS (`services/tts/app/`)**

- *Speech LM on the NIM.* vLLM's completions API takes a list of token ids as the
  prompt and, with `return_token_ids`, streams the generated ids back, so the service's
  prompt layout and token → frame logic didn't change. First audio and RTF matched the
  in-process engine. KV cache pinned at 5 GiB (46,800 tokens).
- *Exact streaming vocoder.* Each chunk is vocoded with 8 frames of left context
  and 4 frames (341 ms) of lookahead. That matches a one-shot decode to ~77 dB
  SNR. Two frames of lookahead would save ~140 ms but leave ~30 dB errors (clicks)
  at chunk edges (`TTS_CONTEXT_RIGHT`).
- *No TF32 in the vocoder.* PyTorch's default TF32 convolutions add a
  shape-dependent ~40 dB noise floor that differs between streamed windows.
  FP32 removes it at the same speed.
- *Adaptive chunks.* The server tracks how much audio the listener has buffered.
  Below 200 ms it sends every frame as soon as it's ready; above that it batches
  up to 4 frames per vocoder call. Concurrent streams' vocoder calls are batched
  together on the GPU.
- Sampling follows the model card's production config (temperature 0.6, top-p 0.9,
  repetition penalty 1.2). Now and then the speech LM doesn't stop and babbles on (we
  saw ~1 in 12 runs of a short punctuated sentence, ~1 in 4 unpunctuated), so generation
  is capped at characters ÷ 9 + 1.2 s per chunk. None of 144 normal clips in six
  languages spoke slower than that; a babbling short sentence now stops at ~5 s instead
  of ~9 s. `TTS_MIN_CHARS_PER_S` changes the 9.

## Operations

**Now (shared models):**

- This repo runs `s2s-agent`, `s2s-web` and `proxy`, which need no GPU. Logs: `docker compose logs -f s2s-agent`.
- The model servers belong to meetmind: `meetmind/deploy/gpu-host` (`asr` Parakeet, `nmt`
  Riva Translate, ~22 GB of GPU memory) and the Magpie NIM. If one is down, calls get no
  captions, translation or speech for that step.
- Restarting meetmind's LiveKit (`meetmind-livekit`) drops calls in both apps. Removing
  this repo's key from its `infra/livekit.yaml` stops this app's calls.
- **Disk:** the own stack's images were removed on 2026-10-27; Docker keeps both the
  compressed and the unpacked copy of each image, so an image takes ~3× its download size.
  `docker builder prune` reclaims build cache after rebuilds.

**Own model stack:**

- **GPU budget** (46 GB L40S): translate 16.9 GB (indic-translate, 3 GiB KV cache), ASR
  NIM 10–12 GB (five fp32 engines), tts-lm 12.4 GB, tts decoder 0.9 GB: 40.3 GB in all,
  ~5.7 GB free. The `dsx-explorer` compose project
  (`/home/ubuntu/DigitalTwin/deploy`) stays stopped.
- **Startup order** is asr → translate → tts-lm → tts → s2s-agent
  (`depends_on: service_healthy`). Each engine sizes its memory from what is free when
  it starts, so they start one at a time.
- **The ASR NIM starts in ~35 s** (it unpacks `custom_model.tar.gz`, 1.6 GB, then warms
  each language). **It restarts itself if Riva dies:** a request over 16 s crashes Riva's
  inference server while the container keeps running, and Docker doesn't restart
  unhealthy containers, so after three failed checks the healthcheck ends the
  container's processes and `restart: unless-stopped` brings it back (measured: 55 s).
- **Adding ASR languages:** accept the checkpoint's terms on Hugging Face, run
  `services/asr-nim/build/build.sh` with all the languages you want (it builds one
  package), add them to `ASR_LANGUAGES` in `services/s2s/app/languages.py` and
  `ASR_WARM_LANGUAGES` in `services/asr-nim/healthcheck.py`, then restart `asr` and
  rebuild `s2s-agent`.
- **Logs:** `docker compose logs -f asr` (or `translate`, `tts-lm`, `tts`, `s2s-agent`).
- **Model files** in `models/` (including `indicconformer-<lang>/`) were downloaded with
  the HF token in `.env`; the token is not passed to the serving containers.

## Licensing

- **Parakeet, Riva Translate 1.6B and Magpie TTS** (in use now) are NVIDIA models, served
  from NVIDIA's Riva and NIM containers on meetmind's servers. NVIDIA's terms for those
  containers apply, as below.
- **NIM containers** (the Riva ASR NIM image, the model-free NIM) are NVIDIA software:
  pulling needs an NGC key, and production use needs an NVIDIA AI Enterprise licence. The
  ASR image is governed by the NVIDIA AI Product Agreement (shown in its startup log).
- **IndicConformer** checkpoints are MIT (AI4Bharat's model cards).
- **indic-translate** and **indic-speak** are under the **Indic Open Model License v1.0** (Bodhan AI / AI4Bharat).
  Self-hosting for your own product is free, with attribution ("Built with … from
  Bodhan AI / AI4Bharat"). Running it as an API that **other companies call directly**
  needs Bodhan AI's written approval, unless you are a nonprofit, government or academic
  user. The license also forbids robocalls and voice cloning of real people without
  consent. It also depends on Llama 3.2 and SNAC, whose licenses apply too.
