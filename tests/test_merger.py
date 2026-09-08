from merger import merge_segments


def test_merge_with_no_speaker_segments_returns_none_speaker():
    transcript = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = merge_segments(transcript, None)
    assert result == [{"start": 0.0, "end": 2.0, "speaker": None, "text": "hello"}]


def test_merge_with_empty_speaker_segments_returns_none_speaker():
    transcript = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    result = merge_segments(transcript, [])
    assert result[0]["speaker"] is None


def test_merge_assigns_matching_speaker():
    transcript = [{"start": 0.0, "end": 2.0, "text": "hello"}]
    speakers = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
    result = merge_segments(transcript, speakers)
    assert result[0]["speaker"] == "SPEAKER_00"


def test_merge_picks_speaker_with_greatest_overlap():
    transcript = [{"start": 4.0, "end": 6.0, "text": "hello"}]
    speakers = [
        {"start": 0.0, "end": 4.5, "speaker": "SPEAKER_00"},
        {"start": 4.5, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    result = merge_segments(transcript, speakers)
    assert result[0]["speaker"] == "SPEAKER_01"


def test_merge_returns_none_speaker_when_no_overlap():
    transcript = [{"start": 10.0, "end": 12.0, "text": "hello"}]
    speakers = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    result = merge_segments(transcript, speakers)
    assert result[0]["speaker"] is None


def test_merge_preserves_segment_order_and_count():
    transcript = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]
    result = merge_segments(transcript, None)
    assert [r["text"] for r in result] == ["a", "b"]
