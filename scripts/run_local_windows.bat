@echo off
rem Builds the local index and runs the evaluation (needs Ollama with qwen2.5:7b and the PDFs in data\).
rem Progress in results\run.log
cd /d "%~dp0.."
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
if not exist results mkdir results
set LOG=results\run.log
echo [%time%] start >> %LOG%
if not exist .venv python -m venv .venv >> %LOG% 2>&1
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt >> %LOG% 2>&1
echo [%time%] ingest >> %LOG%
echo 1/2 Building the index (PDF parsing + embeddings)...
if not exist storage\qdrant_db\collection python ingest.py >> %LOG% 2>&1
echo [%time%] eval >> %LOG%
echo 2/2 Evaluation (18 questions)...
python eval\run_eval.py >> %LOG% 2>&1
echo [%time%] DONE >> %LOG%
echo Done. Results in results\eval.json
pause
