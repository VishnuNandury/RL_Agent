import subprocess
import threading
import time
from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
from api import app as fastapi_app  # your existing FastAPI app

app = FastAPI()

# Mount FastAPI under /api
app.mount("/api", fastapi_app)

# Streamlit runs on port 8501
STREAMLIT_PORT = 8501

# Proxy requests to Streamlit
@app.middleware("http")
async def streamlit_proxy(request: Request, call_next):
    if request.url.path.startswith("/api"):
        return await call_next(request)

    # Proxy to Streamlit
    url = f"http://localhost:{STREAMLIT_PORT}{request.url.path}"
    async with httpx.AsyncClient() as client:
        proxied = await client.request(
            method=request.method,
            url=url,
            headers=request.headers.raw,
            content=await request.body()
        )
        return Response(
            content=proxied.content,
            status_code=proxied.status_code,
            headers=dict(proxied.headers)
        )

def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app2.py", "--server.port=8501"])

if __name__ == "__main__":
    threading.Thread(target=run_streamlit, daemon=True).start()
    time.sleep(2)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
