# mp4文字起こしツール

mp4動画ファイル(音声ファイル wav/mp3 にも対応)をローカル環境で文字起こしするデスクトップツールです。複数話者が存在する場合は話者分離を行い、GUI上で話者ラベルに実名を割り当てられます。出力はTXT/SRT/VTT形式に対応しています。

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

1. 「ファイルを選択」で文字起こししたいファイル(mp4 / wav / mp3)を選ぶ(複数選択可)
2. モデルサイズ・言語・デバイス・話者分離・出力形式・タイムスタンプ表示を設定する
3. 「実行」を押す。処理状況はログ欄に表示される
4. 処理が完了すると結果ウィンドウが開くので、検出された話者(`SPEAKER_00`など)に実名を入力する
5. 「名前を反映してエクスポート」を押すと、入力ファイルと同じフォルダに `<ファイル名>.txt` / `.srt` / `.vtt` が生成される

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
