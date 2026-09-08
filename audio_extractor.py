"""Extract audio from video files using ffmpeg."""
import os
import shutil
import subprocess
import tempfile


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg executable cannot be found on PATH."""


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg fails to extract audio from the input file."""


def extract_audio(video_path: str, output_wav_path: str | None = None) -> str:
    """Extract 16kHz mono WAV audio from a video file using ffmpeg.

    Args:
        video_path: Path to the input video file (e.g. mp4).
        output_wav_path: Where to write the WAV file. If None, a temporary
            file is created and its path returned.

    Returns:
        Path to the extracted WAV file.

    Raises:
        FFmpegNotFoundError: if ffmpeg is not found on PATH.
        AudioExtractionError: if ffmpeg exits with a non-zero status.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "ffmpeg が見つかりません。https://ffmpeg.org/ からインストールし、PATH に追加してください。"
        )

    if output_wav_path is None:
        fd, output_wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        str(output_wav_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioExtractionError(
            f"音声の抽出に失敗しました: {video_path}\n{result.stderr}"
        )

    return output_wav_path
