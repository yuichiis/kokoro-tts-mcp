"""OpenAI TTS-compatible MCP server backed by Kokoro ONNX.

Exposes a ``create_speech`` tool whose arguments mirror OpenAI's
``POST /v1/audio/speech`` endpoint (``input``, ``model``, ``voice``,
``response_format``, ``speed``, ``instructions``) and returns the generated
audio as MCP audio content (base64).

Transport: stdio. Run with ``uv run kokoro-tts-mcp``.

Missing model files are downloaded automatically on first use. To prefetch
everything (model, voices, vocab config and the UniDic dictionary) run
``uv run kokoro-tts-mcp --download``. Downloads are written to
``KOKORO_MODEL_DIR`` / the per-user cache directory, never to stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import soundfile as sf
from mcp.server.mcpserver import MCPServer
from mcp.types import AudioContent
from pydantic import Field

from . import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_FILE = "kokoro-v1.0.onnx"
DEFAULT_VOICES_FILE = "voices-v1.0.bin"
DEFAULT_CONFIG_FILE = "config.json"

MODEL_RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
CONFIG_URL = "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json"
_USER_AGENT = "kokoro-tts-mcp/0.1"

# Default voice for calls that omit `voice`. Set KOKORO_DEFAULT_VOICE to change it.
DEFAULT_VOICE = os.environ.get("KOKORO_DEFAULT_VOICE", "alloy").strip() or "alloy"


@dataclass(frozen=True)
class Asset:
    filename: str
    env_var: str
    url: str
    size: int | None = None


ASSETS: tuple[Asset, ...] = (
    Asset(DEFAULT_MODEL_FILE, "KOKORO_MODEL", f"{MODEL_RELEASE_URL}/{DEFAULT_MODEL_FILE}", 325_505_369),
    Asset(DEFAULT_VOICES_FILE, "KOKORO_VOICES", f"{MODEL_RELEASE_URL}/{DEFAULT_VOICES_FILE}", 28_214_398),
    Asset(DEFAULT_CONFIG_FILE, "KOKORO_CONFIG", CONFIG_URL),
)

# OpenAI voice name -> Kokoro voice id. Kokoro ids can also be passed directly.
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}

RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")

# response_format -> (libsndfile format, subtype, MIME type)
_FORMAT_INFO: dict[str, tuple[str | None, str | None, str]] = {
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/opus"),
    "aac": (None, None, "audio/aac"),
    "flac": ("FLAC", None, "audio/flac"),
    "wav": ("WAV", None, "audio/wav"),
    "pcm": (None, None, "audio/pcm"),
}

# Characters that indicate Japanese text when language="auto".
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
# Sentence boundary used for chunking long input.
_SENTENCE_RE = re.compile(r"[^。．！？!?\n]*[。．！？!?\n]+|[^。．！？!?\n]+$")

_MAX_CHUNK_CHARS = 300
_CHUNK_PAUSE_SECONDS = 0.12

_server = MCPServer(
    name="kokoro-tts-mcp",
    version=__version__,
    instructions=(
        "Local text-to-speech using Kokoro ONNX. "
        "The create_speech tool is compatible with OpenAI's POST /v1/audio/speech: "
        "use input, voice, response_format and speed. "
        "Japanese input is phonemized with misaki+UniDic; English uses the Kokoro tokenizer. "
        "Returns audio content (base64) in the requested format."
    ),
)

_kokoro = None
_ja_g2p = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _log(message: str) -> None:
    """Log to stderr; stdout is reserved for the MCP protocol."""
    print(message, file=sys.stderr, flush=True)


def _auto_download_enabled() -> bool:
    value = os.environ.get("KOKORO_AUTO_DOWNLOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = os.environ.get("XDG_CACHE_HOME")
        root = Path(base) if base else Path.home() / ".cache"
    return root / "kokoro-tts-mcp"


def _asset_target(asset: Asset) -> Path:
    """Where a missing asset should be downloaded to."""
    override = os.environ.get(asset.env_var)
    if override:
        return Path(override).expanduser()
    model_dir = os.environ.get("KOKORO_MODEL_DIR")
    if model_dir:
        return Path(model_dir).expanduser() / asset.filename
    return _cache_dir() / "models" / asset.filename


def _find_file(env_var: str, filename: str, *, required: bool = True) -> Path | None:
    """Locate a model asset via env var, project root, cwd, then the user cache."""
    override = os.environ.get(env_var)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    model_dir = os.environ.get("KOKORO_MODEL_DIR")
    if model_dir:
        candidates.append(Path(model_dir).expanduser() / filename)
    candidates += [
        PROJECT_ROOT / filename,
        Path.cwd() / filename,
        _cache_dir() / "models" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if required:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"{filename} not found. Run 'kokoro-tts-mcp --download' or set {env_var}. "
            f"Tried: {tried}"
        )
    return None


def _asset_ready(asset: Asset) -> Path | None:
    path = _find_file(asset.env_var, asset.filename, required=False)
    if path is None:
        return None
    if asset.size is not None and path.stat().st_size != asset.size:
        return None
    if asset.filename == DEFAULT_CONFIG_FILE:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if "vocab" not in data:
            return None
    return path


def _download_ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle; the Windows store lacks some GitHub intermediates."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(asset: Asset, dest: Path, *, force: bool = False) -> Path:
    if not force and dest.is_file() and (asset.size is None or dest.stat().st_size == asset.size):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    size_label = f"{asset.size / 1048576:.1f} MiB" if asset.size else "unknown size"
    _log(f"Downloading {asset.filename} ({size_label}) -> {dest}")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(asset.url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(
                request, timeout=60, context=_download_ssl_context()
            ) as response, part.open("wb") as file:
                total = int(response.headers.get("Content-Length") or 0) or (asset.size or 0)
                done = 0
                last_report = 0.0
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    file.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 1.0:
                        last_report = now
                        if total:
                            if total >= 1048576:
                                _log(f"  {done / total:6.1%} {done / 1048576:.1f}/{total / 1048576:.1f} MiB")
                            else:
                                _log(f"  {done / total:6.1%} {done}/{total} bytes")
                        else:
                            _log(f"  {done / 1048576:.1f} MiB")

            if asset.size is not None and part.stat().st_size != asset.size:
                raise ValueError(f"expected {asset.size} bytes, got {part.stat().st_size}")
            os.replace(part, dest)
            _log(f"Saved {asset.filename} ({dest.stat().st_size / 1048576:.1f} MiB)")
            return dest
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < 3:
                _log(f"Download failed ({exc}); retrying ({attempt + 1}/3)...")
                time.sleep(attempt * 2)

    raise RuntimeError(f"Failed to download {asset.filename} from {asset.url}: {last_error}")


def _validate_asset(asset: Asset, path: Path) -> None:
    if asset.filename != DEFAULT_CONFIG_FILE:
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Downloaded {asset.filename} is not valid JSON: {exc}") from exc
    if "vocab" not in data:
        raise RuntimeError(f"Downloaded {asset.filename} does not contain a vocabulary.")


def ensure_assets(*, force: bool = False, allow_download: bool | None = None) -> list[Path]:
    """Make sure model files exist, downloading missing or corrupt ones."""
    allow = _auto_download_enabled() if allow_download is None else allow_download
    resolved: list[Path] = []
    for asset in ASSETS:
        ready = None if force else _asset_ready(asset)
        if ready is not None:
            resolved.append(ready)
            continue
        if not allow:
            raise FileNotFoundError(
                f"Missing model file {asset.filename}. "
                "Run 'kokoro-tts-mcp --download' or set KOKORO_AUTO_DOWNLOAD=1."
            )
        existing = _find_file(asset.env_var, asset.filename, required=False)
        path = _download(asset, existing or _asset_target(asset), force=force)
        _validate_asset(asset, path)
        resolved.append(path)
    return resolved


def _unidic_ready() -> bool:
    try:
        import unidic
    except ImportError:
        return False
    return (Path(unidic.__file__).parent / "dicdir" / "mecabrc").is_file()


def ensure_unidic(*, force: bool = False, allow_download: bool | None = None) -> None:
    """Make sure the UniDic dictionary (needed for Japanese G2P) is installed."""
    if not force and _unidic_ready():
        return
    allow = _auto_download_enabled() if allow_download is None else allow_download
    if not allow:
        raise RuntimeError("UniDic dictionary is missing. Run 'uv run python -m unidic download'.")
    _log("Downloading UniDic dictionary (~526 MiB); only needed for Japanese...")
    try:
        subprocess.run(
            [sys.executable, "-m", "unidic", "download"],
            check=True,
            stdout=sys.stderr,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to download UniDic: {exc}") from exc
    if not _unidic_ready():
        raise RuntimeError("UniDic download finished but the dictionary is still missing.")


def _get_kokoro():
    """Load the Kokoro model once, on first use (keeps stdio startup fast)."""
    global _kokoro
    if _kokoro is None:
        with _load_lock:
            if _kokoro is None:
                from kokoro_onnx import Kokoro

                ensure_assets()
                model = _find_file("KOKORO_MODEL", DEFAULT_MODEL_FILE)
                voices = _find_file("KOKORO_VOICES", DEFAULT_VOICES_FILE)
                config = _find_file("KOKORO_CONFIG", DEFAULT_CONFIG_FILE, required=False)
                _kokoro = Kokoro(
                    str(model),
                    str(voices),
                    vocab_config=str(config) if config else None,
                )
    return _kokoro


def _get_ja_g2p():
    global _ja_g2p
    if _ja_g2p is None:
        with _load_lock:
            if _ja_g2p is None:
                ensure_unidic()
                from misaki import ja

                _ja_g2p = ja.JAG2P()
    return _ja_g2p


def _resolve_voice(name: str) -> str:
    key = name.strip().lower()
    voice = OPENAI_VOICE_MAP.get(key, key)
    voices = _get_kokoro().get_voices()
    if voice not in voices:
        raise ValueError(
            f"Unknown voice {name!r}. OpenAI names: {', '.join(OPENAI_VOICE_MAP)}. "
            f"Kokoro voices: {', '.join(voices)}."
        )
    return voice


def _resolve_language(requested: str, text: str, voice: str) -> str:
    if requested != "auto":
        return requested
    if voice.startswith("j"):
        return "ja"
    if _JAPANESE_RE.search(text):
        return "ja"
    return "en"


def _split_text(text: str, limit: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-aligned chunks so long input stays under Kokoro's limit."""
    text = text.strip()
    if not text:
        raise ValueError("input must not be empty")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    buffer = ""
    for sentence in (s for s in _SENTENCE_RE.findall(text) if s):
        while len(sentence) > limit:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.append(sentence[:limit])
            sentence = sentence[limit:]
        if len(buffer) + len(sentence) > limit and buffer:
            chunks.append(buffer)
            buffer = ""
        buffer += sentence
    if buffer:
        chunks.append(buffer)
    return chunks or [text]


def _encode_audio(samples: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    fmt, subtype, mime_type = _FORMAT_INFO[response_format]

    if response_format == "pcm":
        pcm = np.clip(samples, -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes(), mime_type

    if response_format == "aac":
        return _encode_aac(samples, sample_rate), mime_type

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format=fmt, subtype=subtype)
    return buffer.getvalue(), mime_type


def _encode_aac(samples: np.ndarray, sample_rate: int) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("The 'aac' response_format requires ffmpeg on PATH.")
    pcm = np.clip(samples, -1.0, 1.0).astype("<f4").tobytes()
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-f",
            "adts",
            "pipe:1",
        ],
        input=pcm,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to encode AAC: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _synthesize(text: str, voice: str, speed: float, language: str) -> tuple[np.ndarray, int]:
    kokoro = _get_kokoro()
    sample_rate = 24000
    pieces: list[np.ndarray] = []

    with _infer_lock:
        for chunk in _split_text(text):
            if language == "ja":
                phonemes, _ = _get_ja_g2p()(chunk)
                samples, sample_rate = kokoro.create(
                    phonemes, voice=voice, speed=speed, is_phonemes=True
                )
            else:
                samples, sample_rate = kokoro.create(
                    chunk, voice=voice, speed=speed, lang="en-us"
                )
            pieces.append(np.asarray(samples, dtype=np.float32))

    if len(pieces) == 1:
        return pieces[0], sample_rate

    pause = np.zeros(int(sample_rate * _CHUNK_PAUSE_SECONDS), dtype=np.float32)
    merged: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        if index:
            merged.append(pause)
        merged.append(piece)
    return np.concatenate(merged), sample_rate


@_server.tool(
    title="Create speech",
    description=(
        "Generate speech audio from text. Arguments are compatible with OpenAI's "
        "POST /v1/audio/speech. Returns audio content (base64) in the requested format."
    ),
)
def create_speech(
    input: Annotated[
        str,
        Field(min_length=1, max_length=4096, description="Text to synthesize (max 4096 characters)."),
    ],
    model: Annotated[
        Literal["tts-1", "tts-1-hd", "gpt-4o-mini-tts"],
        Field(description="OpenAI model name; accepted for compatibility and ignored."),
    ] = "tts-1",
    voice: Annotated[
        str,
        Field(
            description=(
                "Voice name. OpenAI names (alloy, echo, fable, onyx, nova, shimmer) are "
                "mapped to Kokoro voices; Kokoro ids such as jf_alpha or am_adam also work. "
                "Defaults to KOKORO_DEFAULT_VOICE (or 'alloy')."
            )
        ),
    ] = DEFAULT_VOICE,
    response_format: Annotated[
        Literal["mp3", "opus", "aac", "flac", "wav", "pcm"],
        Field(description="Audio format. 'pcm' is raw 24 kHz signed 16-bit little-endian mono."),
    ] = "mp3",
    speed: Annotated[
        float,
        Field(ge=0.25, le=4.0, description="Playback speed, 0.25 to 4.0."),
    ] = 1.0,
    instructions: Annotated[
        str | None,
        Field(description="Accepted for OpenAI compatibility; not supported by Kokoro."),
    ] = None,
    language: Annotated[
        Literal["auto", "ja", "en"],
        Field(description="G2P language. 'auto' picks Japanese for Japanese voices/text, else English."),
    ] = "auto",
) -> AudioContent:
    """Synthesize speech with Kokoro. Mirrors OpenAI's POST /v1/audio/speech."""
    del model, instructions  # Compatibility-only parameters.

    voice_id = _resolve_voice(voice)
    lang = _resolve_language(language, input, voice_id)
    samples, sample_rate = _synthesize(input, voice_id, speed, lang)
    data, mime_type = _encode_audio(samples, sample_rate, response_format)

    return AudioContent(
        type="audio",
        data=base64.b64encode(data).decode("ascii"),
        mime_type=mime_type,
    )


@_server.tool(
    title="List voices",
    description="List available Kokoro voices and the OpenAI voice-name mapping.",
)
def list_voices() -> str:
    """Return available Kokoro voice ids and the OpenAI-to-Kokoro mapping."""
    kokoro = _get_kokoro()
    payload = {
        "default_voice": DEFAULT_VOICE,
        "openai_to_kokoro": OPENAI_VOICE_MAP,
        "voices": sorted(kokoro.get_voices()),
        "formats": list(RESPONSE_FORMATS),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    # Ensure Japanese text and phonemes survive Windows' default cp932 streams.
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        prog="kokoro-tts-mcp",
        description="Kokoro ONNX MCP server (stdio) with OpenAI-compatible TTS.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model files and the UniDic dictionary, then exit.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="With --download, skip the UniDic dictionary.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download model files even if they already exist.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Directory for model files (sets KOKORO_MODEL_DIR).",
    )
    args = parser.parse_args(argv)

    if args.models_dir:
        os.environ["KOKORO_MODEL_DIR"] = str(args.models_dir.expanduser().resolve())

    if args.download:
        for path in ensure_assets(force=args.force, allow_download=True):
            print(path, flush=True)
        if not args.models_only:
            ensure_unidic(allow_download=True)
        print("All required files are ready.", flush=True)
        return

    _server.run("stdio")


if __name__ == "__main__":
    main()
