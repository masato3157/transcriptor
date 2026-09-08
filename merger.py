"""Merge transcription segments with speaker diarization segments."""


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_segments(transcript_segments, speaker_segments):
    """Assign a speaker label to each transcript segment.

    Args:
        transcript_segments: list of {"start": float, "end": float, "text": str}
        speaker_segments: list of {"start": float, "end": float, "speaker": str},
            or None/empty if speaker diarization was not performed.

    Returns:
        list of {"start": float, "end": float, "speaker": str | None, "text": str}
    """
    merged = []
    for seg in transcript_segments:
        speaker = None
        if speaker_segments:
            best_overlap = 0.0
            for spk_seg in speaker_segments:
                overlap = _overlap(seg["start"], seg["end"], spk_seg["start"], spk_seg["end"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    speaker = spk_seg["speaker"]
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"],
        })
    return merged
