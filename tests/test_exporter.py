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


def test_build_srt_wraps_long_lines_at_25_characters():
    long_text = "あ" * 30  # 30 chars, no speaker -> full line is just the text
    segments = [{"start": 0.0, "end": 2.0, "speaker": None, "text": long_text}]

    result = build_srt(segments, {})

    expected_cue_text = ("あ" * 25) + "\n" + ("あ" * 5)
    assert result == f"1\n00:00:00,000 --> 00:00:02,000\n{expected_cue_text}\n"


def test_build_srt_wrap_counts_speaker_prefix_toward_25_chars():
    # prefix "田中: " is 4 chars, so only 21 chars of body fit on the first line
    body = "あ" * 25
    segments = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": body}]

    result = build_srt(segments, {"SPEAKER_00": "田中"})

    line1 = "田中: " + "あ" * 21
    line2 = "あ" * 4
    assert result == f"1\n00:00:00,000 --> 00:00:02,000\n{line1}\n{line2}\n"


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
