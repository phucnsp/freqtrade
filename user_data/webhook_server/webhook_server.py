# main.py
from fastapi import FastAPI, Form
import logging
from typing import Annotated
import ccxt
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Freqtrade Webhook Receiver")

# Initialize Binance exchange
def get_binance_exchange():
    """Initialize Binance exchange with API credentials"""
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'sandbox': os.getenv('BINANCE_SANDBOX'),
            'enableRateLimit': True,
        })
        return exchange
    except Exception as e:
        logger.error(f"Failed to initialize Binance exchange: {e}")
        return None

async def check_binance_spot_balance():
    """Check Binance spot wallet balance"""
    try:
        exchange = get_binance_exchange()
        if not exchange:
            logger.error("Binance exchange not initialized")
            return None

        balance = exchange.fetch_balance()
        logger.info(f"Binance spot wallet balance: {balance}")

        # Log specific currencies with non-zero balances
        for currency, amounts in balance.items():
            if currency != 'info' and amounts.get('total', 0) > 0:
                logger.info(f"{currency}: Total={amounts['total']}, Free={amounts['free']}, Used={amounts['used']}")

        return balance
    except Exception as e:
        logger.error(f"Error fetching Binance balance: {e}")
        return None

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

    # Check Binance spot balance if this is an entry signal
    if "entry:" in value1.lower():
        logger.info("Entry signal detected - checking Binance spot wallet balance")
        balance = await check_binance_spot_balance()
        if balance:
            logger.info("✅ Successfully retrieved Binance spot balance")
        else:
            logger.warning("❌ Failed to retrieve Binance spot balance")
    else:
        logger.info("No entry signal detected")

    return {"status": "success", "message": "Webhook processed"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
