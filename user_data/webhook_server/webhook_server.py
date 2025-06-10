# main.py
from fastapi import FastAPI, Form
import logging
from typing import Annotated

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Freqtrade Webhook Receiver")

@app.post("/closed_trade")
async def trade_handler(
    value1: Annotated[str, Form()],
    value2: Annotated[str, Form()],
    value3: Annotated[str, Form()]
):
    """
    Core webhook handler for freqtrade form data.
    Receives, parses, and responds with success.
    """
    logging.info("💡 Freqtrade webhook received")

    # Log the webhook data
    logger.info(f"value1: {value1}")
    logger.info(f"value2: {value2}")
    logger.info(f"value3: {value3}")

    return {"status": "success", "message": "Webhook processed"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
