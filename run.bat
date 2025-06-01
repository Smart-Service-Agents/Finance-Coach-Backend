@echo off

cd /d "%~dp0"

start "" cmd /k "python manage.py runserver"