@echo off
title Pakistani Sign Language (PSL) AI System
echo ========================================================
echo Starting Pakistani Sign Language AI Detection System...
echo ========================================================
cd /d "%~dp0backend"
python app\main.py
pause
