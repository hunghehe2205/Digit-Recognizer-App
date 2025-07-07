import multiprocessing
import os
import time
import logging

# Optional: set logging format
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_fastapi():
    """
    Start FastAPI backend (on port 8000)
    """
    import uvicorn
    logger.info("Starting FastAPI backend on port 8000...")
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)


def run_gradio():
    """
    Start Gradio frontend (on port 7860)
    """
    # Add delay to ensure backend starts first
    time.sleep(3)

    from app.front_end.gradio_app import create_interface

    logger.info("Starting Gradio frontend on port 7860...")
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    # Launch FastAPI in a separate process
    backend_process = multiprocessing.Process(target=run_fastapi)
    backend_process.start()

    # Run Gradio in the main process (blocks)
    run_gradio()

    # Optional: wait for backend to end (will only happen on exit)
    backend_process.join()
