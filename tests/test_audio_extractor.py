import os
from unittest.mock import patch, MagicMock

import pytest

from audio_extractor import extract_audio, FFmpegNotFoundError, AudioExtractionError


def test_extract_audio_raises_when_ffmpeg_missing():
    with patch("audio_extractor.shutil.which", return_value=None):
        with pytest.raises(FFmpegNotFoundError):
            extract_audio("input.mp4")


def test_extract_audio_builds_correct_command(tmp_path):
    output_path = tmp_path / "out.wav"
    fake_result = MagicMock(returncode=0, stderr="")
    with patch("audio_extractor.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("audio_extractor.subprocess.run", return_value=fake_result) as mock_run:
        result = extract_audio("input.mp4", output_wav_path=str(output_path))

    assert result == str(output_path)
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert "input.mp4" in args
    assert "16000" in args
    assert str(output_path) in args


def test_extract_audio_raises_on_ffmpeg_failure(tmp_path):
    output_path = tmp_path / "out.wav"
    fake_result = MagicMock(returncode=1, stderr="boom")
    with patch("audio_extractor.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("audio_extractor.subprocess.run", return_value=fake_result):
        with pytest.raises(AudioExtractionError):
            extract_audio("input.mp4", output_wav_path=str(output_path))


def test_extract_audio_creates_temp_file_when_no_output_given():
    fake_result = MagicMock(returncode=0, stderr="")
    with patch("audio_extractor.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("audio_extractor.subprocess.run", return_value=fake_result):
        result = extract_audio("input.mp4")

    assert result.endswith(".wav")
    assert os.path.exists(result)
    os.remove(result)
