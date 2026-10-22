# Live Translator

Microphone or line-in → **NVIDIA Parakeet TDT v3** → **Azure / Gemini / local TranslateGemma** → black window with the translated text, narrated. The sentence stays on screen until its narration finishes, then the next one appears. Every sentence is logged (transcript + translation).

## Install

Python 3.10+ with Tk (included with python.org installers on Windows; `sudo apt install python3-tk` on Debian/Ubuntu).

```
pip install -r requirements.txt
python live_translator.py
```

`requirements.txt` ends with the ONNX Runtime package. **Install only one** of `onnxruntime`, `onnxruntime-directml`, `onnxruntime-gpu`, `onnxruntime-rocm`; they conflict. Do not install `onnx-asr[cpu]` or `[gpu]`, those extras pull in a runtime for you.

The first start downloads the Parakeet model (a couple of GB at full precision; choose `int8` in the Recognition tab for a smaller, CPU-friendly download).

## Acceleration

| Setting | What it uses | Notes |
|---|---|---|
| `auto` (default) | Vulkan CLI if its two paths are filled in, else the best installed ONNX provider (CUDA → DirectML → ROCm → WebGPU), else CPU | |
| `vulkan` | A ggml Vulkan build of Parakeet (see below) | Starts the CLI once per sentence, so each sentence pays the model-load time |
| `directml` | ONNX Runtime + DirectML | AMD, Intel and NVIDIA GPUs on Windows. Model stays loaded, lowest latency for AMD on Windows |
| `rocm` | ONNX Runtime ROCm | AMD on Linux, if your GPU is ROCm-supported |
| `cuda` | ONNX Runtime CUDA | NVIDIA |
| `webgpu` | ONNX Runtime WebGPU | Experimental |
| `cpu` | ONNX Runtime CPU | Works everywhere |

Choosing an acceleration that isn't installed gives an error naming the missing provider instead of silently running on CPU. In `auto` the active backend is written to the log.

### Setting up Vulkan

Python has no Vulkan runtime for Parakeet itself, so this uses a ggml-based C++ port that you build once. Any of these works; fill in the executable, the model file, and pick the matching argument preset in the Recognition tab.

- **whisper.cpp** (`parakeet-cli`): build with `cmake -B build -DGGML_VULKAN=1` then `cmake --build build --target parakeet-cli`. Model: `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` from `ggml-org/parakeet-GGUF`. Args: `-m {model} -f {wav}`.
- **transcribe.cpp** (`transcribe-cli`): Vulkan build per its README. Args: `-m {model} {wav}`.
- **parakeet.cpp** (`parakeet-cli`): Vulkan build per its README. Models: `mudler/parakeet-cpp-gguf`. Args: `transcribe --model {model} --input {wav} --decoder tdt`.

I could not run these binaries here, so the argument presets come from the projects' documentation. If your build uses different flags, edit the args field; `{model}` and `{wav}` are substituted. Use the multilingual **v3** model (v2 is English only).

## Recognition engines

Pick the engine on the **Recognition** tab. Parakeet is the default and the fastest; the others add languages and alternatives.

| Engine | Install | Runs on | Spoken languages |
|---|---|---|---|
| `parakeet` | included | ONNX Runtime: CPU, DirectML, CUDA, ROCm, WebGPU | 25 European languages incl. Hungarian; not Serbian |
| `whisper-onnx` | included (same `onnx-asr`) | same as Parakeet, so **AMD GPUs via DirectML** | about 100 languages incl. Serbian |
| `faster-whisper` | `pip install faster-whisper` | NVIDIA CUDA or CPU only | about 100 languages incl. Serbian |
| `qwen` (Qwen3-ASR) | `pip install -U qwen-asr` plus PyTorch | CUDA/ROCm build of PyTorch, else CPU | 30 languages; of those in this app: Hungarian, English, German, French, Spanish, Italian, Portuguese, Dutch, Romanian, Czech, Polish, Russian, Greek, Swedish, Danish, Finnish, Turkish, Macedonian. **Not Serbian, Croatian, Slovak, Slovenian, Bulgarian** |

- **Whisper and Qwen need the spoken language** set on the General tab for best results; auto-detect on short chunks can guess wrong. The app warns if you choose a language the engine doesn't support.
- **Whisper model**: `onnx-community/whisper-large-v3-turbo` (default for `whisper-onnx`) or `turbo` (`faster-whisper`) give the best quality; `small`/`base` are much faster on weak hardware. `faster-whisper` is greedy-ish with beam size 2 by default (raise it for accuracy). Segments Whisper itself judges as non-speech are dropped, since it tends to invent text on silence.
- **Qwen3-ASR**: `Qwen/Qwen3-ASR-1.7B` is more accurate; `0.6B` is much lighter and is the better pick on CPU. `precision auto` uses bfloat16 on a GPU and float32 on CPU.
- **AMD GPU (your case)**: PyTorch has no GPU support for AMD on Windows, so `qwen` runs on the CPU there, and `faster-whisper` is CPU too. For Whisper on the AMD GPU use `whisper-onnx` with DirectML, or whisper.cpp with Vulkan (below). Qwen3-ASR also exists as GGUF files for the CrispASR runtime; there is a preset for it in the command-line slot, but I could not confirm whether CrispASR has a Vulkan backend, so check its README.
- **Command-line slot** (Vulkan / ggml builds): any ggml CLI works there. `{lang}` is replaced with the spoken-language code, or `auto`. Presets: whisper.cpp `whisper-cli` (`-m {model} -f {wav} -l {lang} -nt -np`) and CrispASR (`-m {model} -f {wav} -l {lang} --backend qwen3`). With `acceleration` set to `vulkan` (or `auto` with the paths filled in) this slot is used instead of the engine above.

## Translators

- **Gemini**: API key from Google AI Studio. Default model is `gemini-3.5-flash-lite` (Google's fastest current model, minimal thinking, best for live use). For better translation quality at more delay choose `gemini-3.6-flash` or `gemini-3.8-flash`. If a model is overloaded (HTTP 503/429/5xx) or no longer exists (404), the app tries the **Fallback models** in order (default `gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.8-flash`) and skips the failing one for a while. The 2.5 models are closed to new Google accounts. Check Google's model page for current names.
- **Azure Translator**: key, region (needed for regional resources) and endpoint (default is the global one). Serbian goes out as `sr-Latn` or `sr-Cyrl`.
- **Local**: any OpenAI-compatible server (llama.cpp `llama-server`, LM Studio, Ollama's `/v1`). If the model name contains `translategemma` the TranslateGemma prompt format is used, otherwise a general translation prompt. A Vulkan build of llama.cpp runs it on an AMD GPU.
- **none**: no translation. The transcript is shown and **read out in the language that was spoken**, using that language's voice (the 'Translate to' choice is ignored). Set the spoken language on the General tab; with `auto` it is detected per sentence if `pip install langdetect` is installed, otherwise the launcher asks you to choose one. A voice picked explicitly on the Narration tab is used for everything.

Keys are stored in the settings JSON in plain text. Alternatively leave the field empty and set `GEMINI_API_KEY` or `AZURE_TRANSLATOR_KEY` (and `AZURE_TRANSLATOR_REGION`).

## Serbian

Target choices are **Serbian (Latin)** and **Serbian (Cyrillic)**. Whatever script the translator returns, the text is converted to the one you picked, so a translator that answers in Cyrillic won't show up in a Latin window. The Edge Serbian voices (`sr-RS-…`) are read from Cyrillic text, so Latin text is converted to Cyrillic only for the voice; the screen still shows Latin. You can turn that off in the Narration tab. The Latin↔Cyrillic converter is rule-based; rare words such as *injekcija* (n+j that are not the letter nj) come out wrong in Cyrillic.

## Client and server

The same program can run in three modes (**Network** tab, or `net_mode` in the settings file):

| Mode | What runs where |
|---|---|
| `standalone` (default) | Everything on one computer, as before |
| `server` | Recognition and translation. Takes audio from clients and sends the recognised and translated sentences back. No microphone or black window needed |
| `client` | Microphone, black window and narration on this computer. The audio goes to the server |

Typical use: the server is the PC with the GPU and the API keys, the client is a laptop at the venue.

**Server.** Pick *server* on the Network tab and press Start, which opens a small log window. Or, with no screen (and no Tk), run `python live_translator.py --server` (optionally `--port`, `--bind`); it uses the settings file, and stops with Ctrl+C. The model is loaded once and shared by all clients (up to 4 at a time); a client that asks for a spoken language the loaded model was not set up for makes the server load a second copy. The server's Recognition and Translation tabs apply; its keys never leave it.

**Client.** Pick *client*, enter the server address and the same port and token, then Start. The client chooses its own spoken language, target language and **Splitting** settings (the threshold is measured on the client's microphone); the engine, models, translator and keys are the server's. Setting the translator to `none` on the client turns translation off for its session (the transcript is then narrated in the spoken language). Narration, voice, speed, the black window, the log and the review files are all on the client. If the connection drops the client reconnects by itself.

**Light client.** For a machine that only needs the microphone and the prompter, use `translator_client.py` instead of the full launcher. It is one small window (server address, token, microphone, languages, narration) and installs just four packages: `pip install -r requirements-client.txt` (numpy, sounddevice, edge-tts, pygame; no ONNX, models or API keys). Copy `translator_client.py` **and** `live_translator.py` into the same folder on the client (it uses the second as a library and loads nothing heavy from it), then run `python translator_client.py`. "Test connection" checks the address and token without starting. Settings are kept in `client_config.json`. To use it without installing Python, `pip install pyinstaller` and `pyinstaller --onefile --noconsole translator_client.py` builds one .exe (I could not test that here).

**Security.** Audio and text travel unencrypted unless you enable TLS, so use it on a trusted network, or encrypt it. A token is required whenever the server listens on anything other than 127.0.0.1. For TLS create a certificate once on the server (replace the address) and copy `server.crt` to the client:

```
openssl req -x509 -newkey rsa:2048 -nodes -keyout server.key -out server.crt -days 365 -subj "/CN=translator" -addext "subjectAltName=IP:192.168.1.50,DNS:translator"
```

On the server set *Certificate file* = `server.crt` and *Private key file* = `server.key`. On the client tick *Use encryption* and set *Server certificate to trust* = `server.crt`, and enter the server address exactly as in the certificate (the IP or `translator`). A VPN such as Tailscale or an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 user@server`, then connect to 127.0.0.1) works as well. To reach the server from outside your network you also have to forward the port on the router, which I would not do without TLS and a long token.

Several clients at once share one log file on the server, so their lines are interleaved; each client's own log and review files are clean.

## Review files

General tab: **Save transcript + translation for review**. Each session writes `review_<date>_<time>_transcript.txt` and `…_translation.txt` to the review folder, with numbered, time-stamped lines (`#12 [21:44:37] …`) so the two files can be read side by side. Without translation only the transcript file is written. In server mode the server writes its own pair per client (the client's address is in the file name) if the option is on there, and the client writes one if it is on there.

## Using it

1. Pick the input device and press **Test input level** while speaking. Set **Splitting → Speech threshold** a few dB below your speaking level and above the room noise.
2. Choose the spoken language and target language, then **Start**. Settings are saved to `live_translator_config.json` next to the script (or the `--config` path).
3. In the black window: **F11 / double-click** full screen, **Esc** leaves it, **+ / −** change the maximum text size. The text is scaled automatically to fill the window. Closing the window stops the session and returns to the launcher.

Logs are in `logs/session_YYYYMMDD_HHMMSS.log`:

```
#3 TRANSCRIPT [hu] (asr 0.41s, audio 3.2s): ...
#3 TRANSLATION [sr-Latn] (0.52s): ...
```

## Splitting the speech into chunks

The **Splitting** tab controls where the audio is cut. Every chunk is recognised, translated and narrated on its own.

| Setting | Effect |
|---|---|
| Pause that ends a chunk | Silence needed to end a chunk. Raise it if sentences are cut in the middle, lower it if text appears too late |
| Ignore sounds shorter than | Filters coughs and clicks |
| Longest chunk | Hard limit for a chunk in normal mode |
| Speech threshold | Level that counts as speech; set it with **Test input level** on the General tab |
| Non-stop speaker mode | See below |
| Only translate complete sentences | Text that ends mid-sentence is held back and joined with the next chunk, so the translator always gets whole sentences. It waits at most the set number of seconds or words, and also sends the held text if the speaker stops talking. Costs some delay |

If the cuts still land in the wrong place, try these in order: raise the pause by 200–300 ms, turn on **Only translate complete sentences**, then adjust the threshold.

## Non-stop speakers

Tick **Splitting → Non-stop speaker mode** for speech that hardly pauses (lectures, broadcasts, interpreting a continuous talk).

- The audio is cut at pauses as usual, and also at the **quietest moment** once a chunk reaches the target length (**Splitting → Target chunk length**, default 5 s, longest 9 s). Nothing is dropped, and the delay stays bounded however long the speaker goes without pausing.
- Narration **speeds up automatically** (+12% per waiting sentence, up to **max extra speed**, default 50%) so it can keep up. Untick **Speak faster when behind** to disable that.
- Up to 8 sentences may queue before the oldest is skipped (still logged as `SKIPPED`).
- **Mute input while narrating** is ignored in this mode, since muting would lose speech.
- Trade-off: chunks are cut mid-sentence, so translation is a bit weaker than in normal mode. Longer chunks translate better but add delay; 6–8 s is a reasonable range. Headphones are strongly recommended.
- The translator and the voice must keep up on average: if the translated Serbian is spoken more slowly than the original, the delay slowly grows, and the app will skip sentences rather than fall further behind.

## Things to know

- **Spoken language**: Parakeet detects the language itself and has no language setting (the General-tab choice only tells the translator what to expect). Whisper and Qwen use the choice as the language to transcribe. Serbian works as a target language with every engine, as a spoken language only with Whisper.
- **Sentence-by-sentence, not word-by-word**: speech is cut into sentences at pauses (**Splitting → Pause that ends a chunk**) and each is transcribed, translated and then narrated, so the text appears when the voice starts, not before. Expect a delay of a few seconds after you stop speaking.
- **Highlighting**: off by default. Tick **Advanced → Highlight the spoken word** for a karaoke-style highlight. With the `edge` voice it follows the voice's word timings when the service provides them, otherwise timing is estimated (as with `pyttsx3`). **Highlight sync offset** shifts it.
- **Falling behind**: if people speak faster than the narration, the oldest waiting sentences are skipped (still logged, marked `SKIPPED`); raise the voice speed or **Max queued sentences**.
- **Feedback**: with speakers and a microphone the narration can be picked up again. Use headphones or tick **Mute input while narrating** (this drops speech during narration).
- Edge TTS and Gemini/Azure need internet; the local translator with `pyttsx3` does not.

## Tested / not tested

Tested here without real hardware (client and server over real local sockets, plain and with TLS, including wrong token, server restart and automatic reconnect, but not across two machines): the launcher (including a 2.5x display scaling check that the Start button stays visible) and black window under a virtual display (auto-fit, resize, highlight timing, fullscreen), the whole pipeline with fake audio/ASR/TTS (sentence cutting, translation, Serbian script handling, logging, shutdown), config save/load, and request formats for the three translators against fakes. **Not tested**: the Whisper and Qwen engines against the real libraries and models (only against stand-ins), real microphones, continuous mode on real speech (only on synthetic audio), Parakeet on your GPU, the Vulkan binaries, live calls to Azure/Gemini/Edge TTS, and playback through real speakers.
