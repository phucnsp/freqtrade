# main.py
import logging
import os
from typing import Annotated
import ccxt
from fastapi import FastAPI, Form

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Freqtrade Webhook Receiver")

# Configuration from environment variables
BOT_CAPITAL_BALANCE = float(os.getenv('BOT_CAPITAL_BALANCE', '3000'))
PROFIT_MIN_THRESHOLD = float(os.getenv('PROFIT_MIN_THRESHOLD', '10'))

logger.info(f"Bot capital balance: {BOT_CAPITAL_BALANCE} USDT")
logger.info(f"Minimum profit threshold: {PROFIT_MIN_THRESHOLD} USDT")
logger.info(f"Will buy ADA if USDT > {BOT_CAPITAL_BALANCE + PROFIT_MIN_THRESHOLD}")

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

async def get_usdt_balance():
    """Get USDT balance from Binance spot wallet"""
    try:
        exchange = get_binance_exchange()
        if not exchange:
            logger.error("Binance exchange not initialized")
            return None, None

        balance = exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        logger.info(f"Current free USDT balance: {usdt_balance}")

        return usdt_balance, exchange
    except Exception as e:
        logger.error(f"Error fetching USDT balance: {e}")
        return None, None

async def buy_ada_with_profit(profit: float, exchange):
    """Buy ADA with the given profit amount"""
    try:
        logger.info(f"💰 Buying ADA with {profit} USDT profit")

        # Get current ADA/USDT price
        ticker = exchange.fetch_ticker('ADA/USDT')
        current_price = ticker['last']
        logger.info(f"Current ADA price: {current_price} USDT")

        # Calculate ADA amount to buy with profit
        ada_amount = round(profit / current_price, 6)

        logger.info(f"Attempting to buy {ada_amount} ADA at limit price {current_price} USDT")

        # Place limit order to buy ADA
        order = exchange.create_limit_buy_order(
            symbol='ADA/USDT',
            amount=ada_amount,
            price=(current_price-0.05)  # Slightly below current price to ensure order fills
        )

        logger.info(f"✅ ADA purchase order placed successfully: {order['id']}")
        logger.info(f"Order details: {ada_amount} ADA at {current_price} USDT")

        return order

    except Exception as e:
        logger.error(f"❌ Error buying ADA with profit: {e}")
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
        logger.info("Entry signal detected - checking USDT balance")
        usdt_balance, exchange = await get_usdt_balance()
        if usdt_balance is not None and exchange is not None:
            logger.info("✅ Successfully retrieved USDT balance")

            profit = usdt_balance - BOT_CAPITAL_BALANCE

            if profit >= PROFIT_MIN_THRESHOLD:
                logger.info(f"💡 Profit ({profit}) >= {PROFIT_MIN_THRESHOLD}, attempting ADA buy")
                ada_order = await buy_ada_with_profit(profit, exchange)
                if ada_order:
                    logger.info("🎉 Successfully placed ADA buy order with profit")
                else:
                    logger.warning("⚠️ Failed to place ADA buy order")
            else:
                logger.info(f"Profit ({profit}) < {PROFIT_MIN_THRESHOLD}, no ADA purchase")

        else:
            logger.warning("❌ Failed to retrieve USDT balance")
    else:
        logger.info("No entry signal detected")

    return {"status": "success", "message": "Webhook processed"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
