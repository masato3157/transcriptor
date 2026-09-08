"""Speaker diarization using pyannote.audio."""
from pyannote.audio import Pipeline


def _resolve_torch_device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def diarize(wav_path: str, hf_token: str, device: str = "auto"):
    """Run speaker diarization on a WAV file.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        hf_token: Hugging Face access token with access to
            pyannote/speaker-diarization-3.1.
        device: "auto", "cpu", or "cuda".

    Returns:
        list of {"start": float, "end": float, "speaker": str}, e.g.
        speaker values like "SPEAKER_00", "SPEAKER_01".
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    pipeline.to(_resolve_torch_device(device))
    diarization = pipeline(wav_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return results
