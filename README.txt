# Alzheimer Chatbot Quickstart

## Run in 3 steps

1. Open terminal in project root

2. Install dependencies

```powershell
pip install -r reqs.txt
```

3. Start Chainlit app

```powershell
python -m chainlit run chabot.py --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

## If `chainlit` command is not recognized

Use the module form (works even when PATH is not set):

```powershell
python -m chainlit run chabot.py --host 127.0.0.1 --port 8000
```

## Optional

- Run Streamlit UI instead:

```powershell
streamlit run chabot.py
```

