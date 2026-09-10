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
    mock_cls.assert_called_once_with("small", device="cpu", compute_type="int8", cpu_threads=4)
    fake_model_instance.transcribe.assert_called_once_with("audio.wav", language="ja")


def test_transcribe_on_cpu_caps_cpu_threads_to_avoid_mkl_allocation_failure():
    # Regression test: loading "large-v3" on CPU with faster-whisper's default
    # thread count (all logical cores) can fail with
    # "RuntimeError: mkl_malloc: failed to allocate memory". Passing a capped
    # cpu_threads avoids the failure (verified manually against the real
    # library). This test locks in that WhisperModel is always constructed
    # with a bounded cpu_threads when running on CPU.
    fake_model_instance = MagicMock()
    fake_model_instance.transcribe.return_value = ([], MagicMock())

    with patch("transcriber.WhisperModel", return_value=fake_model_instance) as mock_cls:
        transcribe("audio.wav", model_size="large-v3", language="ja", device="cpu")

    _, kwargs = mock_cls.call_args
    assert kwargs["cpu_threads"] == 4


def test_transcribe_on_cuda_does_not_pass_cpu_threads():
    fake_model_instance = MagicMock()
    fake_model_instance.transcribe.return_value = ([], MagicMock())

    with patch("transcriber.WhisperModel", return_value=fake_model_instance) as mock_cls:
        transcribe("audio.wav", model_size="small", language="ja", device="cuda")

    mock_cls.assert_called_once_with("small", device="cuda", compute_type="float16")


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
