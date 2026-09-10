"""Build and write transcript output files in TXT, SRT, and VTT formats."""
from pathlib import Path


def format_timestamp_txt(seconds: float) -> str:
    """Format seconds as HH:MM:SS for bracketed TXT timestamps."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _speaker_prefix(segment, speaker_names):
    label = segment.get("speaker")
    if label is None:
        return ""
    name = speaker_names.get(label, label) if speaker_names else label
    return f"{name}: "


SRT_LINE_WIDTH = 25


def _wrap_srt_text(text, width=SRT_LINE_WIDTH):
    """Split text into lines of at most `width` characters each."""
    if not text:
        return [""]
    return [text[i : i + width] for i in range(0, len(text), width)]


def build_txt(segments, speaker_names, include_timestamps):
    """Build TXT content from merged segments."""
    lines = []
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        if include_timestamps:
            ts = format_timestamp_txt(seg["start"])
            lines.append(f"[{ts}] {prefix}{seg['text']}")
        else:
            lines.append(f"{prefix}{seg['text']}")
    return "\n".join(lines) + "\n"


def build_srt(segments, speaker_names):
    """Build SRT content from merged segments."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        prefix = _speaker_prefix(seg, speaker_names)
        start_ts = format_timestamp_srt(seg["start"])
        end_ts = format_timestamp_srt(seg["end"])
        cue_text = "\n".join(_wrap_srt_text(f"{prefix}{seg['text']}"))
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{cue_text}\n")
    return "\n".join(blocks)


def build_vtt(segments, speaker_names):
    """Build WebVTT content from merged segments."""
    blocks = ["WEBVTT\n"]
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        start_ts = format_timestamp_vtt(seg["start"])
        end_ts = format_timestamp_vtt(seg["end"])
        blocks.append(f"{start_ts} --> {end_ts}\n{prefix}{seg['text']}\n")
    return "\n".join(blocks)


_BUILDERS = {
    "srt": lambda segments, speaker_names, include_timestamps: build_srt(segments, speaker_names),
    "vtt": lambda segments, speaker_names, include_timestamps: build_vtt(segments, speaker_names),
    "txt": lambda segments, speaker_names, include_timestamps: build_txt(
        segments, speaker_names, include_timestamps
    ),
}


def export(segments, speaker_names, output_formats, include_timestamps, output_base_path):
    """Write transcript files for each requested format.

    Args:
        segments: merged segments, as returned by merger.merge_segments.
        speaker_names: dict mapping speaker label (e.g. "SPEAKER_00") to a
            real name. May be empty or None if no renaming was done.
        output_formats: iterable of "txt", "srt", "vtt" (any subset).
        include_timestamps: whether TXT output includes bracketed timestamps.
        output_base_path: path (without extension) to write output files to.

    Returns:
        list of file paths written, one per requested format, in the order
        given by output_formats.
    """
    speaker_names = speaker_names or {}
    written = []
    base = Path(output_base_path)
    for fmt in output_formats:
        content = _BUILDERS[fmt](segments, speaker_names, include_timestamps)
        out_path = base.with_name(base.name + f".{fmt}")
        out_path.write_text(content, encoding="utf-8")
        written.append(str(out_path))
    return written
