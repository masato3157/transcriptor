@echo off
cd /d "%~dp0"
echo mp4文字起こしツールを起動しています...
echo (初回起動時はライブラリの読み込みに10〜30秒ほどかかることがあります。しばらくお待ちください)
echo.
python main.py
if errorlevel 1 (
    echo.
    echo エラーが発生しました。上記のメッセージを確認してください。
    pause
)
