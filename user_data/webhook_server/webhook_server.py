# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Freqtrade Webhook Receiver")

class TradePayload(BaseModel):
    value1: str
    value2: str
    value3: str

@app.post("/closed_trade")
async def trade_handler(payload: dict):
    """
    Receives a webhook from Freqtrade when a trade is closed (or entry/entry_cancel).
    Freqtrade will POST JSON like:
      {
        "value1": "Buying BTC/USDT",
        "value2": "limit 50000.00000000",
        "value3": "0.10000000 USDT"
      }
    """
    logging.info("💡 Freqtrade webhook received")
    logging.info(f"Webhook payload: {payload}")
    # logging.info(f"  • value1: {payload.value1}")
    # logging.info(f"  • value2: {payload.value2}")
    # logging.info(f"  • value3: {payload.value3}")
    # data = await request.json()
    # logging.info(f"Received data: {data}")

    # TODO: Add your business logic here, e.g.:
    #  - send a notification
    #  - write to a database
    #  - trigger another service
    #  - etc.

    return {"status": "success", "message": "Webhook processed"}

# Optional: catch all other methods or bad payloads
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logging.error(f"Error handling request: {exc}")
    raise HTTPException(status_code=500, detail="Internal server error")