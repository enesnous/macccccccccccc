@echo off
rem macklip'i baslatir (Windows). Ilk calistirmada gerekenleri kendisi kurar.
cd /d "%~dp0"
where ffmpeg >nul 2>nul || (
  echo ffmpeg bulunamadi. Kurmak icin bir terminalde:  winget install ffmpeg
  pause
  exit /b 1
)
if not exist .venv (
  echo Ilk kurulum yapiliyor, birkac dakika surer...
  python -m venv .venv || (echo Python bulunamadi: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install -q -U pip
  .venv\Scripts\pip install -q -e .
)
.venv\Scripts\macklip-arayuz
pause
