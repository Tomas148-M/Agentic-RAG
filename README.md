# Agentic RAG

První etapa projektu obsahuje základ FastAPI aplikace, konfiguraci a health endpoint.

## Spuštění

Vyžaduje Python 3.10 nebo novější. V kořeni projektu vytvořte a aktivujte virtuální prostředí, poté nainstalujte aplikaci a vývojové závislosti:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Volitelně zkopírujte `.env.example` do `.env` a upravte konfiguraci. Proměnné prostředí mají přednost před hodnotami v `.env`.

```powershell
uvicorn app.main:app --reload
```

Health endpoint je na `http://127.0.0.1:8000/api/v1/health` a interaktivní dokumentace na `http://127.0.0.1:8000/docs`. Endpoint potvrzuje běh API; zatím nekontroluje externí služby.

## Testy

```powershell
python -m pytest
```

Statickou analýzu spusťte pomocí `python -m ruff check .`. Pracovní postup pro implementační úkoly je popsán v `AGENTS.md`.
