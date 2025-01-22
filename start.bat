@echo off
Serving ML App on http://localhost:8000/
waitress-serve ---host=0.0.0.0 --port=8000 app:app
pause
