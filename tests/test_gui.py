from pathlib import Path

from gui import _resolve_output_base_path


def test_resolve_output_base_path_defaults_to_input_folder():
    result = _resolve_output_base_path("D:/videos/meeting.mp4", "")
    assert result == str(Path("D:/videos/meeting.mp4").with_suffix(""))


def test_resolve_output_base_path_uses_specified_output_folder():
    result = _resolve_output_base_path("D:/videos/meeting.mp4", "C:/out")
    assert result == str(Path("C:/out") / "meeting")


def test_resolve_output_base_path_keeps_dotted_filename_intact():
    result = _resolve_output_base_path("D:/videos/2026.09.08 meeting.mp4", "C:/out")
    assert result == str(Path("C:/out") / "2026.09.08 meeting")
