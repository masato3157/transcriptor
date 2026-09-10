"""Speech-to-text transcription using faster-whisper."""
from faster_whisper import WhisperModel


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_cpu_model(model_size: str) -> WhisperModel:
    # faster-whisper defaults cpu_threads to the number of logical cores.
    # For large models (e.g. large-v3), that can make CTranslate2's MKL
    # allocator fail with "mkl_malloc: failed to allocate memory" even
    # with plenty of free RAM. Capping the thread count avoids this.
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)


def _load_model(model_size: str, resolved_device: str) -> WhisperModel:
    if resolved_device != "cuda":
        return _load_cpu_model(model_size)
    try:
        # int8_float16 uses less VRAM than plain float16, which matters on
        # GPUs with little free memory left after other apps.
        return WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    except Exception:
        return _load_cpu_model(model_size)


def transcribe(wav_path: str, model_size: str = "large-v3", language: str | None = "ja", device: str = "auto"):
    """Transcribe a WAV file into timestamped segments.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        model_size: faster-whisper model size, e.g. "small", "medium", "large-v3".
        language: ISO language code (e.g. "ja"), or None for auto-detection.
        device: "auto", "cpu", or "cuda".

    Returns:
        list of {"start": float, "end": float, "text": str}
    """
    resolved_device = _resolve_device(device)
    model = _load_model(model_size, resolved_device)
    segments, _info = model.transcribe(wav_path, language=language)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]
