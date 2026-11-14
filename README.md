# AdaptiLLM 2.0 · Adaptive Edge Intelligence Dashboard

Complete runnable VS Code workspace matching the Colab training pipeline (`SDP_AdaptiLLM.ipynb`) and 6-page dashboard UI.

## Folder Structure
- `app.py` — Complete 6-page Streamlit web application (`Overview`, `AI Playground`, `Resource Monitor`, `Model Management`, `Evaluation`, `Controlled Testing`)
- `requirements.txt` — Python dependencies
- `run_demo.bat` — Windows 11 one-click launcher (avoids PowerShell execution policy errors)
- `artifacts/query_complexity_classifier.joblib` — Pre-trained TF-IDF + Logistic Regression model from your Colab notebook
- `artifacts/metadata.json` — Model metadata & held-out accuracy (`93.3%`)

## Quick Start in VS Code
1. Extract this folder and open `adaptillm_2_demo` in **VS Code**.
2. Double-click `run_demo.bat` (or run `python -m streamlit run app.py` in the VS Code terminal).
3. Your browser will open at `http://localhost:8501`.
