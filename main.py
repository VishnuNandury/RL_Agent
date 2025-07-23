import subprocess
import threading
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from api import app as fastapi_app

app = FastAPI()

# Mount FastAPI app at /api
app.mount("/api", fastapi_app)

# Serve a simple homepage with an iframe pointing to Streamlit UI
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Customer Interaction App</title></head>
        <body>
            <h2>Streamlit Dashboard</h2>
            <iframe src="/iframe" width="100%" height="900" frameborder="0"></iframe>
        </body>
    </html>
    """

@app.get("/iframe", response_class=HTMLResponse)
async def streamlit_iframe():
    return """
    <html>
        <head><title>Streamlit App</title></head>
        <body style="margin:0;padding:0;">
            <iframe src="http://localhost:8501" width="100%" height="1000" frameborder="0"></iframe>
        </body>
    </html>
    """

# Start Streamlit in background
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app2.py", "--server.port=8501"])

if __name__ == "__main__":
    threading.Thread(target=run_streamlit, daemon=True).start()
    time.sleep(2)  # Give Streamlit time to spin up
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
