# ─────────────────────────────────────────────────────────────────────────────
# SMC_ML_Strategy.py
# Estrategia híbrida: Smart Money Concepts + Filtro ML
# Exchange: Bybit | Par: BTC/USDT | Timeframe: 15m
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import ta

class SMC_ML_Strategy(IStrategy):
    """
    Capa 1: Detección de estructura SMC en 15m
    Capa 2: Bias direccional desde 1h (HTF)
    Capa 3: Filtro ML (próxima iteración)
    """

    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = False

    minimal_roi = {
        "0":  0.03,
        "30": 0.02,
        "60": 0.01,
        "120": 0
    }
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_20_1h'] = ta.trend.ema_indicator(dataframe['close'], window=20)
        dataframe['ema_50_1h'] = ta.trend.ema_indicator(dataframe['close'], window=50)
        dataframe['ema_200_1h'] = ta.trend.ema_indicator(dataframe['close'], window=200)

        dataframe['htf_bias'] = 0
        dataframe.loc[
            (dataframe['ema_20_1h'] > dataframe['ema_50_1h']) &
            (dataframe['ema_50_1h'] > dataframe['ema_200_1h']),
            'htf_bias'
        ] = 1
        dataframe.loc[
            (dataframe['ema_20_1h'] < dataframe['ema_50_1h']) &
            (dataframe['ema_50_1h'] < dataframe['ema_200_1h']),
            'htf_bias'
        ] = -1

        dataframe['atr_1h'] = ta.volatility.average_true_range(
            dataframe['high'], dataframe['low'], dataframe['close'], window=14
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_9']   = ta.trend.ema_indicator(dataframe['close'], window=9)
        dataframe['ema_21']  = ta.trend.ema_indicator(dataframe['close'], window=21)
        dataframe['ema_50']  = ta.trend.ema_indicator(dataframe['close'], window=50)
        dataframe['ema_200'] = ta.trend.ema_indicator(dataframe['close'], window=200)

        dataframe['rsi'] = ta.momentum.rsi(dataframe['close'], window=14)
        macd = ta.trend.MACD(dataframe['close'])
        dataframe['macd']        = macd.macd()
        dataframe['macd_signal'] = macd.macd_signal()
        dataframe['macd_hist']   = macd.macd_diff()

        dataframe['atr'] = ta.volatility.average_true_range(
            dataframe['high'], dataframe['low'], dataframe['close'], window=14
        )
        bb = ta.volatility.BollingerBands(dataframe['close'], window=20, window_dev=2)
        dataframe['bb_upper']  = bb.bollinger_hband()
        dataframe['bb_lower']  = bb.bollinger_lband()
        dataframe['bb_mid']    = bb.bollinger_mavg()
        dataframe['bb_width']  = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']

        dataframe['volume_ma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma_20']

        dataframe = self._detect_swing_points(dataframe)
        dataframe = self._detect_order_blocks(dataframe)
        dataframe = self._detect_fvg(dataframe)
        dataframe = self._detect_liquidity(dataframe)
        dataframe = self._detect_kill_zone(dataframe)
        dataframe = self._detect_bos(dataframe)

        return dataframe

    def _detect_swing_points(self, df: DataFrame, lookback: int = 5) -> DataFrame:
        df['swing_high'] = (
            df['high'] == df['high'].rolling(lookback * 2 + 1, center=True).max()
        ).astype(int)
        df['swing_low'] = (
            df['low'] == df['low'].rolling(lookback * 2 + 1, center=True).min()
        ).astype(int)
        df['last_swing_high'] = df['high'].where(df['swing_high'] == 1).ffill()
        df['last_swing_low']  = df['low'].where(df['swing_low'] == 1).ffill()
        return df

    def _detect_bos(self, df: DataFrame) -> DataFrame:
        df['bos_bullish'] = (
            (df['close'] > df['last_swing_high'].shift(1)) &
            (df['close'].shift(1) <= df['last_swing_high'].shift(1))
        ).astype(int)
        df['bos_bearish'] = (
            (df['close'] < df['last_swing_low'].shift(1)) &
            (df['close'].shift(1) >= df['last_swing_low'].shift(1))
        ).astype(int)
        return df

    def _detect_order_blocks(self, df: DataFrame, lookback: int = 3) -> DataFrame:
        body = abs(df['close'] - df['open'])
        avg_body = body.rolling(20).mean()
        impulse = body > avg_body * 1.5

        df['ob_bullish'] = (
            (df['close'] < df['open']) &
            impulse.shift(-1) &
            (df['close'].shift(-1) > df['open'].shift(-1))
        ).astype(int)
        df['ob_bull_top']    = df['open'].where(df['ob_bullish'] == 1).ffill()
        df['ob_bull_bottom'] = df['close'].where(df['ob_bullish'] == 1).ffill()

        df['ob_bearish'] = (
            (df['close'] > df['open']) &
            impulse.shift(-1) &
            (df['close'].shift(-1) < df['open'].shift(-1))
        ).astype(int)
        df['ob_bear_top']    = df['close'].where(df['ob_bearish'] == 1).ffill()
        df['ob_bear_bottom'] = df['open'].where(df['ob_bearish'] == 1).ffill()
        return df

    def _detect_fvg(self, df: DataFrame) -> DataFrame:
        df['fvg_bullish'] = (
            df['high'].shift(1) < df['low'].shift(-1)
        ).astype(int)
        df['fvg_bull_bottom'] = df['high'].shift(1).where(df['fvg_bullish'] == 1).ffill()
        df['fvg_bull_top']    = df['low'].shift(-1).where(df['fvg_bullish'] == 1).ffill()
        df['fvg_bearish'] = (
            df['low'].shift(1) > df['high'].shift(-1)
        ).astype(int)
        return df

    def _detect_liquidity(self, df: DataFrame, tolerance: float = 0.001) -> DataFrame:
        df['liquidity_high'] = (
            (df['high'].rolling(20).max() - df['high']).abs() / df['high'] < tolerance
        ).astype(int)
        df['liquidity_low'] = (
            (df['low'].rolling(20).min() - df['low']).abs() / df['low'] < tolerance
        ).astype(int)
        return df

    def _detect_kill_zone(self, df: DataFrame) -> DataFrame:
        hour = df['date'].dt.hour
        df['in_kill_zone'] = (
            ((hour >= 7) & (hour < 10)) |
            ((hour >= 12) & (hour < 15))
        ).astype(int)
        df['london_open'] = ((hour >= 7) & (hour < 10)).astype(int)
        df['ny_open']     = ((hour >= 12) & (hour < 15)).astype(int)
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['htf_bias_1h'] == 1) &
                (dataframe['bos_bullish'] == 1) &
                (
                    (dataframe['close'] <= dataframe['ob_bull_top']) |
                    (dataframe['close'] <= dataframe['fvg_bull_top'])
                ) &
                (dataframe['rsi'] < 70) &
                (dataframe['rsi'] > 30) &
                (dataframe['in_kill_zone'] == 1) &
                (dataframe['volume_ratio'] > 1.2) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bos_bearish'] == 1) |
                (dataframe['rsi'] > 80) |
                (dataframe['htf_bias_1h'] == -1)
            ),
            'exit_long'
        ] = 1
        return dataframe