"""Speech-to-text transcription using faster-whisper."""
from faster_whisper import WhisperModel


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    compute_type = "float16" if resolved_device == "cuda" else "int8"
    if resolved_device == "cpu":
        # faster-whisper defaults cpu_threads to the number of logical cores.
        # For large models (e.g. large-v3), that can make CTranslate2's MKL
        # allocator fail with "mkl_malloc: failed to allocate memory" even
        # with plenty of free RAM. Capping the thread count avoids this.
        model = WhisperModel(
            model_size, device=resolved_device, compute_type=compute_type, cpu_threads=4
        )
    else:
        model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
    segments, _info = model.transcribe(wav_path, language=language)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]
