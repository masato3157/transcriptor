from unittest.mock import patch, MagicMock

from diarizer import diarize


def test_diarize_returns_speaker_segments():
    fake_turn_1 = MagicMock(start=0.0, end=2.0)
    fake_turn_2 = MagicMock(start=2.0, end=4.0)
    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (fake_turn_1, None, "SPEAKER_00"),
        (fake_turn_2, None, "SPEAKER_01"),
    ]
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = fake_diarization

    with patch("diarizer.Pipeline.from_pretrained", return_value=fake_pipeline) as mock_from_pretrained:
        result = diarize("audio.wav", hf_token="fake-token", device="cpu")

    mock_from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", use_auth_token="fake-token"
    )
    fake_pipeline.assert_called_once_with("audio.wav")
    assert result == [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]
