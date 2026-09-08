# mp4文字起こしツール Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** mp4動画をローカルのWhisperで文字起こしし、複数話者を検出してGUI上で実名を割り当て、TXT/SRT/VTT形式で出力するデスクトップツールを作る。

**Architecture:** ffmpegで音声抽出 → faster-whisperで文字起こし → pyannote.audioで話者分離(任意) → 時間重複マッチングでセグメントに話者ラベルを付与 → Tkinter GUIで結果表示・話者名入力 → TXT/SRT/VTTへエクスポート。各処理段階を独立した純粋関数/モジュールとして実装し、GUIはそれらを呼び出すだけの薄い層にする。

**Tech Stack:** Python 3.10, faster-whisper, pyannote.audio, torch, python-dotenv, tkinter(標準ライブラリ), pytest

**Spec:** [docs/superpowers/specs/2026-09-08-mp4-transcription-tool-design.md](../specs/2026-09-08-mp4-transcription-tool-design.md)

## Global Constraints

- 実行環境: Python 3.10、ffmpeg(PATH上に導入済み)、(任意)NVIDIA GPU + CUDA。
- デフォルトモデルサイズ: `large-v3`。デフォルト言語: 日本語固定(`ja`)。デフォルトデバイス: 自動検出。
- デフォルト出力形式: TXT・SRT(VTTはデフォルトOFF、チェックボックスで有効化)。TXTのタイムスタンプ表示はデフォルトON。
- 話者分離はデフォルトON。HuggingFaceトークンは環境変数`HF_TOKEN`(`.env`経由)から読み込む。
- 出力ファイルは入力mp4と同じフォルダに `<元ファイル名>.txt` / `.srt` / `.vtt` として保存する。
- 話者分離の話者数指定・手動区間修正、クラウドAPI対応、リアルタイム文字起こしはスコープ外。

---

### Task 1: プロジェクトの初期セットアップ

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: なし
- Produces: 以降のタスクが利用する依存パッケージ一覧、環境変数`HF_TOKEN`の設定例、テストディレクトリ構造。

- [ ] **Step 1: requirements.txt を作成**

```
faster-whisper>=1.0.3
pyannote.audio>=3.1.1
torch>=2.1.0
python-dotenv>=1.0.1
pytest>=8.0.0
```

- [ ] **Step 2: .env.example を作成**

```
# Hugging Face アクセストークン(pyannote/speaker-diarization-3.1 の利用権限が必要)
# 1. https://huggingface.co/settings/tokens でトークンを発行
# 2. https://huggingface.co/pyannote/speaker-diarization-3.1 の利用規約に同意
# 3. 下記に取得したトークンを設定し、ファイル名を .env にリネームする
HF_TOKEN=your_token_here
```

- [ ] **Step 3: .gitignore を作成**

```
__pycache__/
*.pyc
.env
*.wav
venv/
.venv/
.pytest_cache/
```

- [ ] **Step 4: テスト用の空パッケージファイルを作成**

`tests/__init__.py` を空ファイルとして作成する。

- [ ] **Step 5: 依存パッケージをインストールして動作確認**

Run: `pip install -r requirements.txt`
Expected: エラーなくインストールが完了する(torch/pyannote.audioのダウンロードに数分かかる場合がある)。

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example .gitignore tests/__init__.py
git commit -m "chore: add project scaffolding (deps, env example, gitignore)"
```

---

### Task 2: audio_extractor.py — ffmpegによる音声抽出

**Files:**
- Create: `audio_extractor.py`
- Test: `tests/test_audio_extractor.py`

**Interfaces:**
- Consumes: なし(標準ライブラリの`shutil`, `subprocess`, `tempfile`のみ)
- Produces: `extract_audio(video_path: str, output_wav_path: str | None = None) -> str` — mp4等の動画から16kHz mono WAVを抽出しファイルパスを返す。`output_wav_path`未指定時は一時ファイルを作成する。例外`FFmpegNotFoundError`(ffmpeg未検出)、`AudioExtractionError`(抽出失敗)を送出する。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_audio_extractor.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_audio_extractor.py -v`
Expected: FAIL(`audio_extractor`モジュールが存在しないため`ModuleNotFoundError`)

- [ ] **Step 3: audio_extractor.py を実装**

```python
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_audio_extractor.py -v`
Expected: PASS(4件全て)

- [ ] **Step 5: Commit**

```bash
git add audio_extractor.py tests/test_audio_extractor.py
git commit -m "feat: add ffmpeg-based audio extraction"
```

---

### Task 3: merger.py — 文字起こしセグメントと話者区間のマージ

**Files:**
- Create: `merger.py`
- Test: `tests/test_merger.py`

**Interfaces:**
- Consumes: なし
- Produces: `merge_segments(transcript_segments: list[dict], speaker_segments: list[dict] | None) -> list[dict]`
  - 入力`transcript_segments`要素: `{"start": float, "end": float, "text": str}`
  - 入力`speaker_segments`要素: `{"start": float, "end": float, "speaker": str}`。`None`または空リストの場合は話者分離なし扱い。
  - 出力要素: `{"start": float, "end": float, "speaker": str | None, "text": str}`。各文字起こしセグメントに対し、時間の重なりが最大の話者区間のラベルを割り当てる。重なりがなければ`speaker=None`。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_merger.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_merger.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'merger'`)

- [ ] **Step 3: merger.py を実装**

```python
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_merger.py -v`
Expected: PASS(6件全て)

- [ ] **Step 5: Commit**

```bash
git add merger.py tests/test_merger.py
git commit -m "feat: add transcript/speaker segment merging logic"
```

---

### Task 4: exporter.py — TXT/SRT/VTT出力

**Files:**
- Create: `exporter.py`
- Test: `tests/test_exporter.py`

**Interfaces:**
- Consumes: `merger.merge_segments`が返す形式のセグメントリスト(`{"start", "end", "speaker", "text"}`)
- Produces:
  - `format_timestamp_txt(seconds: float) -> str` — `"HH:MM:SS"`
  - `format_timestamp_srt(seconds: float) -> str` — `"HH:MM:SS,mmm"`
  - `format_timestamp_vtt(seconds: float) -> str` — `"HH:MM:SS.mmm"`
  - `build_txt(segments, speaker_names, include_timestamps) -> str`
  - `build_srt(segments, speaker_names) -> str`
  - `build_vtt(segments, speaker_names) -> str`
  - `export(segments, speaker_names, output_formats, include_timestamps, output_base_path) -> list[str]` — 指定形式(`"txt"`/`"srt"`/`"vtt"`の任意の組み合わせ)でファイルを書き出し、書き込んだファイルパスのリストを返す。`speaker_names`は`{話者ラベル: 実名}`の辞書(`None`または空でも可、その場合はラベルをそのまま表示)。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_exporter.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_exporter.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'exporter'`)

- [ ] **Step 3: exporter.py を実装**

```python
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
        blocks.append(f"{i}\n{start_ts} --> {end_ts}\n{prefix}{seg['text']}\n")
    return "\n".join(blocks) + "\n"


def build_vtt(segments, speaker_names):
    """Build WebVTT content from merged segments."""
    blocks = ["WEBVTT\n"]
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        start_ts = format_timestamp_vtt(seg["start"])
        end_ts = format_timestamp_vtt(seg["end"])
        blocks.append(f"{start_ts} --> {end_ts}\n{prefix}{seg['text']}\n")
    return "\n".join(blocks) + "\n"


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
        out_path = base.with_suffix(f".{fmt}")
        out_path.write_text(content, encoding="utf-8")
        written.append(str(out_path))
    return written
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_exporter.py -v`
Expected: PASS(11件全て)

- [ ] **Step 5: Commit**

```bash
git add exporter.py tests/test_exporter.py
git commit -m "feat: add TXT/SRT/VTT export with speaker name substitution"
```

---

### Task 5: transcriber.py — faster-whisperによる文字起こし

**Files:**
- Create: `transcriber.py`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `faster_whisper.WhisperModel`(外部ライブラリ)
- Produces: `transcribe(wav_path: str, model_size: str = "large-v3", language: str | None = "ja", device: str = "auto") -> list[dict]` — `[{"start": float, "end": float, "text": str}, ...]`を返す。`merger.merge_segments`の`transcript_segments`引数として利用される。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_transcriber.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'transcriber'`)

- [ ] **Step 3: transcriber.py を実装**

```python
"""Speech-to-text transcription using faster-whisper."""
from faster_whisper import WhisperModel


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe(wav_path: str, model_size: str = "large-v3", language: str | None = "ja", device: str = "auto"):
    """Transcribe a WAV file into timestamped segments.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        model_size: faster-whisper model size, e.g. "small", "medium", "large-v3".
        language: ISO language code (e.g. "ja"), or None for auto-detection.
        device: "auto", "cpu", or "cuda".

    Returns:
        list of {"start": float, "end": float, "text": str}
    """
    resolved_device = _resolve_device(device)
    compute_type = "float16" if resolved_device == "cuda" else "int8"
    model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
    segments, _info = model.transcribe(wav_path, language=language)
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_transcriber.py -v`
Expected: PASS(4件全て)

- [ ] **Step 5: Commit**

```bash
git add transcriber.py tests/test_transcriber.py
git commit -m "feat: add faster-whisper transcription wrapper"
```

---

### Task 6: diarizer.py — pyannote.audioによる話者分離

**Files:**
- Create: `diarizer.py`
- Test: `tests/test_diarizer.py`

**Interfaces:**
- Consumes: `pyannote.audio.Pipeline`(外部ライブラリ)
- Produces: `diarize(wav_path: str, hf_token: str, device: str = "auto") -> list[dict]` — `[{"start": float, "end": float, "speaker": str}, ...]`(例: `speaker="SPEAKER_00"`)を返す。`merger.merge_segments`の`speaker_segments`引数として利用される。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_diarizer.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_diarizer.py -v`
Expected: FAIL(`ModuleNotFoundError: No module named 'diarizer'`)

- [ ] **Step 3: diarizer.py を実装**

```python
"""Speaker diarization using pyannote.audio."""
from pyannote.audio import Pipeline


def _resolve_torch_device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def diarize(wav_path: str, hf_token: str, device: str = "auto"):
    """Run speaker diarization on a WAV file.

    Args:
        wav_path: Path to a 16kHz mono WAV file.
        hf_token: Hugging Face access token with access to
            pyannote/speaker-diarization-3.1.
        device: "auto", "cpu", or "cuda".

    Returns:
        list of {"start": float, "end": float, "speaker": str}, e.g.
        speaker values like "SPEAKER_00", "SPEAKER_01".
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    pipeline.to(_resolve_torch_device(device))
    diarization = pipeline(wav_path)

    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return results
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_diarizer.py -v`
Expected: PASS(1件)

- [ ] **Step 5: Commit**

```bash
git add diarizer.py tests/test_diarizer.py
git commit -m "feat: add pyannote.audio speaker diarization wrapper"
```

---

### Task 7: gui.py / main.py — Tkinter GUIと処理の統合

**Files:**
- Create: `gui.py`
- Create: `main.py`

**Interfaces:**
- Consumes: `audio_extractor.extract_audio`, `transcriber.transcribe`, `diarizer.diarize`, `merger.merge_segments`, `exporter.export`, `exporter.format_timestamp_txt`(結果一覧のプレビュー表示用)
- Produces: `run_app()` — Tkinterアプリを起動するエントリ関数。`main.py`から呼び出される。

このタスクはTkinter GUIであり、Task 1〜6のような自動テストは対象外とする(spec「テスト方針」に準拠)。Step 4で手動確認を行う。

- [ ] **Step 1: gui.py を実装**

```python
"""Tkinter GUI for the mp4 transcription tool."""
import os
import threading
from pathlib import Path
from tkinter import (
    Tk, Toplevel, Frame, Label, Entry, Button, Listbox, Text, Checkbutton,
    StringVar, BooleanVar, END, DISABLED, NORMAL, filedialog, messagebox, ttk,
)

from dotenv import load_dotenv

import audio_extractor
import transcriber
import diarizer
import merger
import exporter

load_dotenv()

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGE_OPTIONS = {"日本語": "ja", "自動検出": None}
DEVICE_OPTIONS = {"自動": "auto", "CPU": "cpu", "GPU": "cuda"}


class TranscriptionApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("mp4文字起こしツール")
        self.geometry("700x550")

        self.selected_files = []

        self._build_main_screen()

    def _build_main_screen(self):
        file_frame = Frame(self)
        file_frame.pack(fill="x", padx=10, pady=5)
        Button(file_frame, text="ファイルを選択", command=self._select_files).pack(side="left")

        self.file_listbox = Listbox(self, height=5)
        self.file_listbox.pack(fill="x", padx=10, pady=5)

        settings_frame = Frame(self)
        settings_frame.pack(fill="x", padx=10, pady=5)

        Label(settings_frame, text="モデルサイズ:").grid(row=0, column=0, sticky="w")
        self.model_size_var = StringVar(value="large-v3")
        ttk.Combobox(
            settings_frame, textvariable=self.model_size_var, values=MODEL_SIZES, state="readonly"
        ).grid(row=0, column=1, sticky="w")

        Label(settings_frame, text="言語:").grid(row=0, column=2, sticky="w")
        self.language_var = StringVar(value="日本語")
        ttk.Combobox(
            settings_frame, textvariable=self.language_var,
            values=list(LANGUAGE_OPTIONS.keys()), state="readonly"
        ).grid(row=0, column=3, sticky="w")

        Label(settings_frame, text="デバイス:").grid(row=1, column=0, sticky="w")
        self.device_var = StringVar(value="自動")
        ttk.Combobox(
            settings_frame, textvariable=self.device_var,
            values=list(DEVICE_OPTIONS.keys()), state="readonly"
        ).grid(row=1, column=1, sticky="w")

        self.diarization_var = BooleanVar(value=True)
        Checkbutton(
            settings_frame, text="話者分離を行う", variable=self.diarization_var
        ).grid(row=1, column=2, columnspan=2, sticky="w")

        self.format_txt_var = BooleanVar(value=True)
        self.format_srt_var = BooleanVar(value=True)
        self.format_vtt_var = BooleanVar(value=False)
        Checkbutton(settings_frame, text="TXT", variable=self.format_txt_var).grid(row=2, column=0, sticky="w")
        Checkbutton(settings_frame, text="SRT", variable=self.format_srt_var).grid(row=2, column=1, sticky="w")
        Checkbutton(settings_frame, text="VTT", variable=self.format_vtt_var).grid(row=2, column=2, sticky="w")

        self.include_timestamps_var = BooleanVar(value=True)
        Checkbutton(
            settings_frame, text="TXT出力にタイムスタンプを含める",
            variable=self.include_timestamps_var
        ).grid(row=3, column=0, columnspan=3, sticky="w")

        self.run_button = Button(self, text="実行", command=self._on_run_clicked)
        self.run_button.pack(pady=5)

        self.log_text = Text(self, height=15, state=DISABLED)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

    def _select_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("MP4 動画", "*.mp4")])
        if not paths:
            return
        self.selected_files = list(paths)
        self.file_listbox.delete(0, END)
        for p in self.selected_files:
            self.file_listbox.insert(END, p)

    def log(self, message):
        def _append():
            self.log_text.configure(state=NORMAL)
            self.log_text.insert(END, message + "\n")
            self.log_text.see(END)
            self.log_text.configure(state=DISABLED)
        self.after(0, _append)

    def _on_run_clicked(self):
        if not self.selected_files:
            messagebox.showwarning("ファイル未選択", "mp4ファイルを選択してください。")
            return

        output_formats = []
        if self.format_txt_var.get():
            output_formats.append("txt")
        if self.format_srt_var.get():
            output_formats.append("srt")
        if self.format_vtt_var.get():
            output_formats.append("vtt")
        if not output_formats:
            messagebox.showwarning("出力形式未選択", "出力形式を1つ以上選択してください。")
            return

        diarization_enabled = self.diarization_var.get()
        hf_token = os.environ.get("HF_TOKEN")
        if diarization_enabled and not hf_token:
            proceed = messagebox.askyesno(
                "HuggingFaceトークン未設定",
                "話者分離に必要なHF_TOKENが設定されていません。話者分離なしで続行しますか？",
            )
            if not proceed:
                return
            diarization_enabled = False

        settings = {
            "model_size": self.model_size_var.get(),
            "language": LANGUAGE_OPTIONS[self.language_var.get()],
            "device": DEVICE_OPTIONS[self.device_var.get()],
            "diarization_enabled": diarization_enabled,
            "hf_token": hf_token,
            "output_formats": output_formats,
            "include_timestamps": self.include_timestamps_var.get(),
        }

        self.run_button.configure(state=DISABLED)
        files = list(self.selected_files)
        thread = threading.Thread(target=self._run_pipeline, args=(files, settings), daemon=True)
        thread.start()

    def _run_pipeline(self, files, settings):
        for video_path in files:
            try:
                self._process_file(video_path, settings)
            except Exception as exc:
                self.log(f"[{video_path}] エラー: {exc}")
        self.after(0, lambda: self.run_button.configure(state=NORMAL))

    def _process_file(self, video_path, settings):
        self.log(f"[{video_path}] 音声抽出中...")
        wav_path = audio_extractor.extract_audio(video_path)
        try:
            self.log(f"[{video_path}] 文字起こし中...")
            transcript_segments = transcriber.transcribe(
                wav_path,
                model_size=settings["model_size"],
                language=settings["language"],
                device=settings["device"],
            )

            speaker_segments = None
            if settings["diarization_enabled"]:
                self.log(f"[{video_path}] 話者分離中...")
                try:
                    speaker_segments = diarizer.diarize(
                        wav_path, hf_token=settings["hf_token"], device=settings["device"]
                    )
                except Exception as exc:
                    self.log(f"[{video_path}] 話者分離に失敗しました: {exc}")
                    proceed = self._confirm_on_main_thread(
                        "話者分離に失敗しました。話者分離なしで続行しますか？"
                    )
                    if not proceed:
                        self.log(f"[{video_path}] 処理を中止しました。")
                        return
                    speaker_segments = None
        finally:
            os.remove(wav_path)

        merged = merger.merge_segments(transcript_segments, speaker_segments)
        self.after(0, lambda: self._show_results_window(video_path, merged, settings))

    def _confirm_on_main_thread(self, message):
        result = {}
        event = threading.Event()

        def ask():
            result["value"] = messagebox.askyesno("確認", message)
            event.set()

        self.after(0, ask)
        event.wait()
        return result.get("value", False)

    def _show_results_window(self, video_path, merged_segments, settings):
        window = Toplevel(self)
        window.title(f"結果: {Path(video_path).name}")
        window.geometry("600x550")

        text_widget = Text(window, height=15)
        text_widget.pack(fill="both", expand=True, padx=10, pady=5)
        for seg in merged_segments:
            label = seg["speaker"] or ""
            ts = exporter.format_timestamp_txt(seg["start"])
            text_widget.insert(END, f"[{ts}] {label} {seg['text']}\n")
        text_widget.configure(state=DISABLED)

        speaker_labels = sorted({seg["speaker"] for seg in merged_segments if seg["speaker"]})
        name_vars = {}
        if speaker_labels:
            names_frame = Frame(window)
            names_frame.pack(fill="x", padx=10, pady=5)
            Label(names_frame, text="話者名の入力:").grid(row=0, column=0, columnspan=2, sticky="w")
            for i, label in enumerate(speaker_labels, start=1):
                Label(names_frame, text=label).grid(row=i, column=0, sticky="w")
                var = StringVar(value=label)
                Entry(names_frame, textvariable=var).grid(row=i, column=1, sticky="w")
                name_vars[label] = var

        def on_export():
            speaker_names = {label: var.get() for label, var in name_vars.items()}
            base_path = str(Path(video_path).with_suffix(""))
            written = exporter.export(
                merged_segments,
                speaker_names,
                settings["output_formats"],
                settings["include_timestamps"],
                base_path,
            )
            self.log(f"[{video_path}] 出力完了: {', '.join(written)}")
            messagebox.showinfo("完了", "エクスポートが完了しました:\n" + "\n".join(written))

        Button(window, text="名前を反映してエクスポート", command=on_export).pack(pady=5)


def run_app():
    app = TranscriptionApp()
    app.mainloop()
```

- [ ] **Step 2: main.py を実装**

```python
"""Entry point for the mp4 transcription tool."""
from gui import run_app

if __name__ == "__main__":
    run_app()
```

- [ ] **Step 3: 起動確認(構文・インポートエラーがないことの確認)**

Run: `python -c "import gui"`
Expected: エラーなく終了する(GUIウィンドウは開かない)。

- [ ] **Step 4: 手動でGUIを起動して基本フローを確認**

Run: `python main.py`

確認項目:
- ウィンドウが起動し、「ファイルを選択」ボタンでmp4ファイルを選択できる。
- 出力形式チェックボックス(TXT/SRT/VTT)、話者分離トグル、タイムスタンプ表示トグルが操作できる。
- 短いmp4ファイル(数秒〜数十秒、日本語音声、可能なら2話者)で「実行」を押し、ログ欄に処理状況が表示される。
- 処理完了後、結果ウィンドウが開き、セグメント一覧と話者名入力欄(話者分離ONの場合)が表示される。
- 話者名を入力して「名前を反映してエクスポート」を押すと、入力mp4と同じフォルダに指定した形式のファイルが生成され、内容に実名が反映されている。
- HF_TOKEN未設定の状態で話者分離ONのまま実行すると確認ダイアログが表示され、「いいえ」でキャンセルできる。

- [ ] **Step 5: Commit**

```bash
git add gui.py main.py
git commit -m "feat: add Tkinter GUI and application entry point"
```

---

### Task 8: README.md の作成と結合動作の最終確認

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: なし
- Produces: セットアップ手順・使い方ドキュメント。

- [ ] **Step 1: README.md を作成**

```markdown
# mp4文字起こしツール

mp4動画ファイルをローカル環境で文字起こしするデスクトップツールです。複数話者が存在する動画では話者分離を行い、GUI上で話者ラベルに実名を割り当てられます。出力はTXT/SRT/VTT形式に対応しています。

## セットアップ

### 1. 前提ソフトウェア

- Python 3.10以上
- [ffmpeg](https://ffmpeg.org/) がインストール済みで、PATHに追加されていること

確認方法:

```bash
python --version
ffmpeg -version
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

torch や pyannote.audio のダウンロードには数分かかる場合があります。NVIDIA GPUがある場合は、CUDA対応版のtorchを別途インストールするとGPUで高速に処理できます([PyTorch公式](https://pytorch.org/get-started/locally/)を参照)。

### 3. Hugging Faceトークンの取得(話者分離を使う場合)

話者分離機能(pyannote.audio)を使うには、Hugging Faceのアクセストークンが必要です。

1. [Hugging Face](https://huggingface.co/join) の無料アカウントを作成する
2. [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) にアクセスし、利用規約に同意する
3. [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) でアクセストークンを発行する
4. `.env.example` を `.env` にコピーし、`HF_TOKEN=` の後に発行したトークンを貼り付ける

話者分離を使わない場合、この手順は不要です(GUI上のトグルでOFFにできます)。

## 使い方

```bash
python main.py
```

1. 「ファイルを選択」で文字起こししたいmp4ファイルを選ぶ(複数選択可)
2. モデルサイズ・言語・デバイス・話者分離・出力形式・タイムスタンプ表示を設定する
3. 「実行」を押す。処理状況はログ欄に表示される
4. 処理が完了すると結果ウィンドウが開くので、検出された話者(`SPEAKER_00`など)に実名を入力する
5. 「名前を反映してエクスポート」を押すと、入力mp4と同じフォルダに `<ファイル名>.txt` / `.srt` / `.vtt` が生成される

## 出力例

```
[00:00:05] 田中: おはようございます。
[00:00:08] 鈴木: よろしくお願いします。
```

## テストの実行

```bash
pytest -v
```

## トラブルシューティング

- 「ffmpegが見つかりません」と表示される: ffmpegをインストールし、PATHに追加してください。
- 話者分離が失敗する: `.env` の `HF_TOKEN` が正しいか、モデルの利用規約に同意済みか確認してください。GUI上で「話者分離なしで続行」を選ぶこともできます。
- 処理が遅い: モデルサイズを `small` や `medium` に下げる、またはGPU(CUDA)を使うと高速化できます。
```

- [ ] **Step 2: 全自動テストを実行して回帰がないことを確認**

Run: `pytest -v`
Expected: PASS(Task 2〜6で作成した全テストが通る)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add setup and usage instructions"
```

---

## Self-Review Notes

- **spec網羅性:** 概要・GUI仕様・出力フォーマット(TXT/SRT/VTT・タイムスタンプ有無)・話者分離・エラー処理(ffmpeg未検出/非対応ファイル/モデルDL失敗/HFトークン未設定/話者分離失敗)・依存パッケージ・テスト方針は、それぞれTask 1〜8でカバーされている。GUIの「複数ファイルを一括選択した場合の順次処理」はTask 7の`_run_pipeline`のforループで対応済み。
- **プレースホルダ:** 各タスクのコードは完全な実装であり、TODOやスタブは含まない。
- **型・シグネチャの整合性:** `merger.merge_segments`の出力形式(`{"start","end","speaker","text"}`)が`exporter`の各関数・`gui.py`の`_show_results_window`/`on_export`で一貫して使われていることを確認済み。`transcriber.transcribe`と`diarizer.diarize`の戻り値形式も`merger.merge_segments`の入力仕様と一致している。
