# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
strategy:
+ Buy ADA if wallet balance higher than 3000 USDT
+ only update available_capital in config if needed

References:
1. ElliotV5HO.py
2. execution logic in live mode: https://www.freqtrade.io/en/latest/bot-basics/#bot-execution-logic
"""
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from functools import reduce
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import logging

logger = logging.getLogger(__name__)

class DoNothingStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # minimal_roi = {"0": 0.05}  # {"60": 0.01, "30": 0.02, "0": 0.04}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -1.0  # stoploss -100% means ignore stoploss

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = (
        False # only exit with signal, since we are trading with good coin, can hold longer
    )
    exit_profit_only = True  # if buy and then price down and sell triggered, no exit, continue hold
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = True  # priority to entry signal

    max_open_trades = 1 # set 1 so that all available_capital will go ALL, to customer_stake_amount, so that we can adjust it all as we want
    available_capital = (
        1000  # each bot only allow to use this amount, even the wallet has more than that
    )

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 0  # 79  # need to do recursive analysis for define this value

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # Momentum Indicators
        # ------------------------------------

        # ADX
        # dataframe["adx"] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe["plus_dm"] = ta.PLUS_DM(dataframe)
        # dataframe["plus_di"] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe["minus_dm"] = ta.MINUS_DM(dataframe)
        # dataframe["minus_di"] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe["aroonup"] = aroon["aroonup"]
        # dataframe["aroondown"] = aroon["aroondown"]
        # dataframe["aroonosc"] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe["ao"] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe["uo"] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe["cci"] = ta.CCI(dataframe)

        # RSI
        # dataframe["rsi"] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe["rsi"] - 50)
        # dataframe["fisher_rsi"] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe["fisher_rsi_norma"] = 50 * (dataframe["fisher_rsi"] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe["slowd"] = stoch["slowd"]
        # dataframe["slowk"] = stoch["slowk"]

        # Stochastic Fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe["fastd"] = stoch_fast["fastd"]
        # dataframe["fastk"] = stoch_fast["fastk"]

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe["fastd_rsi"] = stoch_rsi["fastd"]
        # dataframe["fastk_rsi"] = stoch_rsi["fastk"]

        # MACD
        # macd = ta.MACD(dataframe)
        # dataframe["macd"] = macd["macd"]
        # dataframe["macdsignal"] = macd["macdsignal"]
        # dataframe["macdhist"] = macd["macdhist"]

        # MFI
        # dataframe["mfi"] = ta.MFI(dataframe)

        # # ROC
        # dataframe["roc"] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        # dataframe["bb_lowerband"] = bollinger["lower"]
        # dataframe["bb_middleband"] = bollinger["mid"]
        # dataframe["bb_upperband"] = bollinger["upper"]
        # dataframe["bb_percent"] = (
        #     (dataframe["close"] - dataframe["bb_lowerband"]) /
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        # )
        # dataframe["bb_width"] = (
        #     (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        # )

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) / dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe["ema3"] = ta.EMA(dataframe, timeperiod=3)
        # dataframe["ema5"] = ta.EMA(dataframe, timeperiod=5)
        # dataframe["ema10"] = ta.EMA(dataframe, timeperiod=10)
        # dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        # dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        # dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe["sma3"] = ta.SMA(dataframe, timeperiod=3)
        # dataframe["sma5"] = ta.SMA(dataframe, timeperiod=5)
        # dataframe["sma10"] = ta.SMA(dataframe, timeperiod=10)
        # dataframe["sma21"] = ta.SMA(dataframe, timeperiod=21)
        # dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
        # dataframe["sma100"] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        # dataframe["sar"] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        # dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe["htsine"] = hilbert["sine"]
        # dataframe["htleadsine"] = hilbert["leadsine"]

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe["CDLHAMMER"] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe["CDLINVERTEDHAMMER"] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe["CDLDRAGONFLYDOJI"] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe["CDLPIERCING"] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe["CDLMORNINGSTAR"] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe["CDL3WHITESOLDIERS"] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe["CDLHANGINGMAN"] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe["CDLSHOOTINGSTAR"] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe["CDLGRAVESTONEDOJI"] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe["CDLDARKCLOUDCOVER"] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe["CDLEVENINGDOJISTAR"] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe["CDLEVENINGSTAR"] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe["CDL3LINESTRIKE"] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe["CDLSPINNINGTOP"] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe["CDLENGULFING"] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe["CDLHARAMI"] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe["CDL3OUTSIDE"] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe["CDL3INSIDE"] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe["ha_open"] = heikinashi["open"]
        # dataframe["ha_close"] = heikinashi["close"]
        # dataframe["ha_high"] = heikinashi["high"]
        # dataframe["ha_low"] = heikinashi["low"]

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ("live", "dry_run"):
                ob = self.dp.orderbook(metadata["pair"], 1)
                dataframe["best_bid"] = ob["bids"][0][0]
                dataframe["best_ask"] = ob["asks"][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        return dataframe

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply does nothing.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass

    def custom_entry_price(
        self,
        pair: str,
        trade: Trade | None,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Custom entry price logic, returning the new entry price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set entry price

        :param pair: Pair that's currently analyzed
        :param trade: trade object (None for initial entries).
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided
        """
        return proposed_rate

    def adjust_order_price(
        self,
        trade: Trade,
        order: Order | None,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        current_order_rate: float,
        entry_tag: str | None,
        side: str,
        is_entry: bool,
        **kwargs,
    ) -> float:
        """
        Exit and entry order price re-adjustment logic, returning the user desired limit price.
        This only executes when a order was already placed, still open (unfilled fully or partially)
        and not timed out on subsequent candles after entry trigger.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-callbacks/

        When not implemented by a strategy, returns current_order_rate as default.
        If current_order_rate is returned then the existing order is maintained.
        If None is returned then order gets canceled but not replaced by a new one.

        :param pair: Pair that's currently analyzed
        :param trade: Trade object.
        :param order: Order object
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in entry_pricing.
        :param current_order_rate: Rate of the existing order in place.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param is_entry: True if the order is an entry order, False if it's an exit order.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New entry price value if provided

        """
        return current_order_rate

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ) -> float:
        """
        Custom exit price logic, returning the new exit price.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None, orderbook is used to set exit price

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param proposed_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param exit_tag: Exit reason.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New exit price value if provided
        """
        return proposed_rate

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize stake size for each new trade.

        Examples:
        def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: float | None, max_stake: float,
                                leverage: float, entry_tag: str | None, side: str,
                                **kwargs) -> float:

            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()

            if current_candle["fastk_rsi_1h"] > current_candle["fastd_rsi_1h"]:
                if self.config["stake_amount"] == "unlimited":
                    # Use entire available wallet during favorable conditions when in compounding mode.
                    return max_stake
                else:
                    # Compound profits during favorable conditions instead of using a static stake.
                    return self.wallets.get_total_stake_amount() / self.config["max_open_trades"]

            # Use default stake amount.
            return proposed_stake

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param leverage: Leverage selected for this trade.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A stake size, which is between min_stake and max_stake.
        """

        return proposed_stake

    use_custom_stoploss = True

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value.
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param after_fill: True if the stoploss is called after the order was filled.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current_rate
        """

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """
        Custom exit signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting sell signal on a candle at specified
        time. This method is not called when sell signal is set.

        This method should be overridden to create sell signals that depend on trade parameters. For
        example you could implement a sell relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom exit reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute sell, return a string with custom exit reason or True. Otherwise return
        None or False.
        """
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                            'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        return True

    def check_entry_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        Check entry timeout function callback.
        This method can be used to override the entry-timeout.
        It is called whenever a limit entry order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the entry order is cancelled.
        """
        return False

    def check_exit_timeout(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> bool:
        """
        Check exit timeout function callback.
        This method can be used to override the exit-timeout.
        It is called whenever a limit exit order has been created,
        and is not yet fully filled.
        Configuration options in `unfilledtimeout` will be verified before this,
        so ensure to set these timeouts high enough.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, this simply returns False.
        :param pair: Pair the trade is for
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the exit-order is cancelled.
        """
        return False

    position_adjustment_enable = True
    max_entry_position_adjustment = (
        1000000  # number of additional entries per trade, can use hyperopt to tune it
    )

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra entry or exit orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current entry rate (same as current_entry_profit)
        :param current_profit: Current profit (as ratio), calculated based on current_rate
                                (same as current_entry_profit).
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        Example:
            def adjust_trade_position():
                if trade.has_open_orders:
                    # Only act if no orders are open
                    return

                if current_profit > 0.05 and trade.nr_of_successful_exits == 0:
                    # Take half of the profit at +5%
                    return -(trade.stake_amount / 2), "half_profit_5%"

                if current_profit > -0.05:
                    return None

                # Obtain pair dataframe (just to show how to access it)
                dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                # Only buy when not actively falling price.
                last_candle = dataframe.iloc[-1].squeeze()
                previous_candle = dataframe.iloc[-2].squeeze()
                if last_candle["close"] < previous_candle["close"]:
                    return None

                filled_entries = trade.select_filled_orders(trade.entry_side)
                count_of_entries = trade.nr_of_successful_entries
                # Allow up to 3 additional increasingly larger buys (4 in total)
                # Initial buy is 1x
                # If that falls to -5% profit, we buy 1.25x more, average profit should increase to roughly -2.2%
                # If that falls down to -5% again, we buy 1.5x more
                # If that falls once again down to -5%, we buy 1.75x more
                # Total stake for this trade would be 1 + 1.25 + 1.5 + 1.75 = 5.5x of the initial allowed stake.
                # That is why max_dca_multiplier is 5.5
                # Hope you have a deep wallet!
                try:
                    # This returns first order stake size
                    stake_amount = filled_entries[0].stake_amount_filled
                    # This then calculates current safety order size
                    stake_amount = stake_amount * (1 + (count_of_entries * 0.25))
                    return stake_amount, "1/3rd_increase"
                except Exception as exception:
                    return None

                return None
        """

        # If available_capital have more than 10 USDT from its original amount in the wallet, we will accumulate it in ADA
        # only update available_capital in config if needed
        total_usdt = self.wallets.get_total('USDT')
        if total_usdt > (10+10): #(self.available_capital + 10):
            logger.info(f"Available capital: {self.available_capital}, total USDT: {total_usdt}")
            stake_amount = total_usdt - 10 #self.available_capital
            logger.info(f"Accumulating {stake_amount} USDT in ADA")
            return stake_amount, "accumulate_ada"
        logger.info(f"total USDT: {total_usdt} is not enough to accumulate in ADA")
        return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 1.0

    def order_filled(
        self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs
    ) -> None:
        """
        Called right after an order fills.
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: Trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        pass
