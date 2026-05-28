from fastapi import FastAPI


app = FastAPI(title="Mnemosyne Agent API", version="0.1.0")

@app.post("/health")
def health():
    return {"status": "healthy"}
