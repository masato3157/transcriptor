from unittest.mock import patch, MagicMock

from transcriber import transcribe, _resolve_device


def _make_fake_segment(start, end, text):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def test_transcribe_returns_segments_as_dicts():
    fake_segments = [
        _make_fake_segment(0.0, 1.5, " こんにちは "),
        _make_fake_segment(1.5, 3.0, " さようなら "),
    ]
    fake_model_instance = MagicMock()
    fake_model_instance.transcribe.return_value = (fake_segments, MagicMock())

    with patch("transcriber.WhisperModel", return_value=fake_model_instance) as mock_cls:
        result = transcribe("audio.wav", model_size="small", language="ja", device="cpu")

    assert result == [
        {"start": 0.0, "end": 1.5, "text": "こんにちは"},
        {"start": 1.5, "end": 3.0, "text": "さようなら"},
    ]
    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8")
    fake_model_instance.transcribe.assert_called_once_with("audio.wav", language="ja")


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _resolve_device("auto") == "cpu"


def test_resolve_device_auto_uses_cuda_when_available():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": fake_torch}):
        assert _resolve_device("auto") == "cuda"


def test_resolve_device_passthrough_when_not_auto():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
