"""Tkinter GUI for the mp4 transcription tool."""
import os
import platform
import subprocess
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


def _resolve_output_base_path(video_path, output_folder):
    """Return the path (without extension) exporter.export should write to.

    If output_folder is empty, files are written next to video_path (the
    original behavior). Otherwise, they're written into output_folder using
    video_path's base filename.
    """
    stem = Path(video_path).stem
    if output_folder:
        return str(Path(output_folder) / stem)
    return str(Path(video_path).with_suffix(""))


def _open_folder(folder_path):
    """Open the given folder in the OS's file explorer, cross-platform."""
    system = platform.system()
    if system == "Windows":
        os.startfile(folder_path)
    elif system == "Darwin":
        subprocess.run(["open", folder_path])
    else:
        subprocess.run(["xdg-open", folder_path])


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

        output_folder_frame = Frame(self)
        output_folder_frame.pack(fill="x", padx=10, pady=5)
        Label(output_folder_frame, text="保存先フォルダ:").pack(side="left")
        self.output_folder_var = StringVar(value="")
        Entry(output_folder_frame, textvariable=self.output_folder_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=5
        )
        Button(output_folder_frame, text="参照", command=self._select_output_folder).pack(side="left")
        Label(
            output_folder_frame, text="(未指定なら入力ファイルと同じフォルダ)", fg="gray"
        ).pack(side="left", padx=5)

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
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("対応ファイル", "*.mp4 *.wav *.mp3"),
                ("MP4 動画", "*.mp4"),
                ("WAV 音声", "*.wav"),
                ("MP3 音声", "*.mp3"),
            ]
        )
        if not paths:
            return
        self.selected_files = list(paths)
        self.file_listbox.delete(0, END)
        for p in self.selected_files:
            self.file_listbox.insert(END, p)

    def _select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_var.set(folder)

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
            "output_folder": self.output_folder_var.get(),
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

        base_path = _resolve_output_base_path(video_path, settings["output_folder"])
        written = exporter.export(
            merged, {}, settings["output_formats"], settings["include_timestamps"], base_path
        )
        self.log(f"[{video_path}] 自動保存しました: {', '.join(written)}")

        self.after(0, lambda: self._show_results_window(video_path, merged, settings, base_path))

    def _confirm_on_main_thread(self, message):
        result = {}
        event = threading.Event()

        def ask():
            result["value"] = messagebox.askyesno("確認", message)
            event.set()

        self.after(0, ask)
        event.wait()
        return result.get("value", False)

    def _show_results_window(self, video_path, merged_segments, settings, base_path):
        window = Toplevel(self)
        window.title(f"結果: {Path(video_path).name}")
        window.geometry("600x550")

        Label(
            window,
            text=(
                "話者ラベル(SPEAKER_00等)のまま既に自動保存済みです。"
                "実名を入力して「名前を反映してエクスポート」を押すと、同じファイルに上書き保存されます。"
                "このウィンドウを閉じても保存済みの内容は失われません。"
            ),
            wraplength=560,
            justify="left",
        ).pack(fill="x", padx=10, pady=5)

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

        open_folder_button = {"widget": None}

        def on_export():
            speaker_names = {label: var.get() for label, var in name_vars.items()}
            written = exporter.export(
                merged_segments,
                speaker_names,
                settings["output_formats"],
                settings["include_timestamps"],
                base_path,
            )
            self.log(f"[{video_path}] 実名を反映して上書き保存しました: {', '.join(written)}")
            messagebox.showinfo("完了", "エクスポートが完了しました:\n" + "\n".join(written))

            output_folder = str(Path(base_path).parent)
            if open_folder_button["widget"] is None:
                open_folder_button["widget"] = Button(
                    window,
                    text="出力先フォルダを開く",
                    command=lambda: _open_folder(output_folder),
                )
                open_folder_button["widget"].pack(pady=5)

        Button(window, text="名前を反映してエクスポート", command=on_export).pack(pady=5)


def run_app():
    app = TranscriptionApp()
    app.mainloop()
