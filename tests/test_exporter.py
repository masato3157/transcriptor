from pathlib import Path

from exporter import (
    format_timestamp_txt,
    format_timestamp_srt,
    format_timestamp_vtt,
    build_txt,
    build_srt,
    build_vtt,
    export,
)


SEGMENTS_WITH_SPEAKERS = [
    {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00", "text": "おはようございます。"},
    {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "よろしくお願いします。"},
]

SEGMENTS_NO_SPEAKERS = [
    {"start": 5.0, "end": 7.0, "speaker": None, "text": "おはようございます。"},
]

SPEAKER_NAMES = {"SPEAKER_00": "田中", "SPEAKER_01": "鈴木"}


def test_format_timestamp_txt():
    assert format_timestamp_txt(5.0) == "00:00:05"
    assert format_timestamp_txt(3665.0) == "01:01:05"


def test_format_timestamp_srt():
    assert format_timestamp_srt(5.123) == "00:00:05,123"


def test_format_timestamp_vtt():
    assert format_timestamp_vtt(5.123) == "00:00:05.123"


def test_build_txt_with_timestamps_and_speakers():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, include_timestamps=True)
    assert result == (
        "[00:00:05] 田中: おはようございます。\n"
        "[00:00:08] 鈴木: よろしくお願いします。\n"
    )


def test_build_txt_without_timestamps():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, include_timestamps=False)
    assert result == (
        "田中: おはようございます。\n"
        "鈴木: よろしくお願いします。\n"
    )


def test_build_txt_without_speakers():
    result = build_txt(SEGMENTS_NO_SPEAKERS, {}, include_timestamps=False)
    assert result == "おはようございます。\n"


def test_build_txt_falls_back_to_label_when_name_missing():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, {}, include_timestamps=False)
    assert result == (
        "SPEAKER_00: おはようございます。\n"
        "SPEAKER_01: よろしくお願いします。\n"
    )


def test_build_srt_format():
    result = build_srt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES)
    assert result == (
        "1\n00:00:05,000 --> 00:00:07,000\n田中: おはようございます。\n\n"
        "2\n00:00:08,000 --> 00:00:10,000\n鈴木: よろしくお願いします。\n"
    )


def test_build_srt_splits_long_text_into_multiple_cues_at_word_boundaries():
    # "あ"*30 tokenizes into 15 "ああ" tokens (2 chars each, no particles), so
    # greedy packing to width=25 lands on a token boundary at 24 chars, not a
    # mid-token cut at 25.
    long_text = "あ" * 30
    segments = [{"start": 0.0, "end": 3.0, "speaker": None, "text": long_text}]

    result = build_srt(segments, {})

    line1 = "あ" * 24
    line2 = "あ" * 6
    # Cue duration is split proportionally by character count: 24/30 and 6/30
    # of the 3.0s segment duration.
    assert result == (
        "1\n00:00:00,000 --> 00:00:02,400\n" + line1 + "\n\n"
        "2\n00:00:02,400 --> 00:00:03,000\n" + line2 + "\n"
    )


def test_build_srt_never_splits_a_single_token_mid_word():
    # A single unbroken token longer than the width (here, a run of latin
    # letters Janome treats as one noun token) still gets hard-split, since
    # there's no word boundary to break on within it.
    long_text = "A" * 30
    segments = [{"start": 0.0, "end": 3.0, "speaker": None, "text": long_text}]

    result = build_srt(segments, {})

    line1 = "A" * 25
    line2 = "A" * 5
    assert result == (
        "1\n00:00:00,000 --> 00:00:02,500\n" + line1 + "\n\n"
        "2\n00:00:02,500 --> 00:00:03,000\n" + line2 + "\n"
    )


def test_build_srt_prefers_breaking_after_a_particle_before_reaching_the_width_limit():
    # Real hallucination-free example: breaking at a particle (助詞) once the
    # line has reached a minimum length reads more naturally than packing
    # all the way to the 25-character limit.
    text = "ですが規則でこれ以上のことは申し上げられないんです。"
    segments = [{"start": 0.0, "end": 14.06, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    line1 = "ですが規則でこれ以上の"
    line2 = "ことは申し上げられないんです。"
    assert result.count("-->") == 2
    assert line1 in result
    assert line2 in result
    assert result.index(line1) < result.index(line2)


def test_build_srt_wrap_counts_speaker_prefix_toward_the_limit():
    # prefix "田中: " (4 chars) counts toward the first line's budget, so
    # fewer body tokens fit before the same token-boundary wrap kicks in.
    body = "あ" * 25
    segments = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": body}]

    result = build_srt(segments, {"SPEAKER_00": "田中"})

    line1 = "田中: " + "あ" * 20  # 4-char prefix + 20 chars = 24 (next token
    # would make it 26 > 25, so it wraps at the token boundary before that)
    line2 = "あ" * 5
    assert result == (
        f"1\n00:00:00,000 --> 00:00:01,655\n{line1}\n\n"
        f"2\n00:00:01,655 --> 00:00:02,000\n{line2}\n"
    )


def test_build_srt_does_not_wrap_short_lines():
    segments = [{"start": 0.0, "end": 2.0, "speaker": None, "text": "短い文"}]
    result = build_srt(segments, {})
    assert result == "1\n00:00:00,000 --> 00:00:02,000\n短い文\n"


def test_build_vtt_format():
    result = build_vtt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES)
    assert result == (
        "WEBVTT\n\n"
        "00:00:05.000 --> 00:00:07.000\n田中: おはようございます。\n\n"
        "00:00:08.000 --> 00:00:10.000\n鈴木: よろしくお願いします。\n"
    )


def test_export_writes_requested_formats(tmp_path):
    base_path = tmp_path / "video"
    written = export(
        SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt", "srt", "vtt"], True, str(base_path)
    )

    assert written == [
        str(base_path.with_suffix(".txt")),
        str(base_path.with_suffix(".srt")),
        str(base_path.with_suffix(".vtt")),
    ]
    for path in written:
        assert Path(path).exists()
    assert "田中" in Path(written[0]).read_text(encoding="utf-8")


def test_export_only_writes_selected_formats(tmp_path):
    base_path = tmp_path / "video"
    written = export(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt"], True, str(base_path))

    assert written == [str(base_path.with_suffix(".txt"))]
    assert not base_path.with_suffix(".srt").exists()


def test_export_handles_dotted_base_filename(tmp_path):
    base_path = tmp_path / "2026.09.08 meeting"
    written = export(
        SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt", "srt", "vtt"], True, str(base_path)
    )

    assert written == [
        str(tmp_path / "2026.09.08 meeting.txt"),
        str(tmp_path / "2026.09.08 meeting.srt"),
        str(tmp_path / "2026.09.08 meeting.vtt"),
    ]
    for path in written:
        assert Path(path).exists()
