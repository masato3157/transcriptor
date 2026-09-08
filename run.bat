@echo off
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo エラーが発生しました。上記のメッセージを確認してください。
    pause
)
