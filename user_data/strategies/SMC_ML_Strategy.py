# ─────────────────────────────────────────────────────────────────────────────
# SMC_ML_Strategy.py — Anti-Leakage v3 — Feature alignment fix
# Exchange: Bybit | Par: BTC/USDT | Timeframe: 15m
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy, informative
import ta
import joblib
import os

class SMC_ML_Strategy(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = False

    minimal_roi = {
        "0":  0.02,
        "15": 0.01,
        "30": 0.005,
        "60": 0
    }
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    startup_candle_count = 200
    process_only_new_candles = True

    ml_score_threshold = 0.60
    ml_model = None
    ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'smc_ml_model.pkl')

    # ── Fuente única de verdad — idéntica a train_model.py ───────────────────
    FEATURE_COLS = [
        'htf_bias_1h', 'htf_ema20_50_dist_1h',
        'htf_ema50_200_dist_1h', 'htf_price_ema200_dist_1h',
        'dist_to_ob', 'ob_size', 'ob_age',
        'dist_to_fvg', 'fvg_size',
        'dist_to_swing_high', 'dist_to_swing_low',
        'volume_ratio', 'volume_zscore',
        'atr_norm', 'atr_ratio', 'bb_width',
        'rsi', 'macd_hist',
        'in_kill_zone', 'bos_bullish',
    ]

    def bot_start(self, **kwargs):
        if os.path.exists(self.ml_model_path):
            self.ml_model = joblib.load(self.ml_model_path)
            print(f"✅ Modelo ML cargado: {self.ml_model_path}")
        else:
            print("⚠️  Modelo ML no encontrado. Operando sin filtro ML.")

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_20_1h']  = ta.trend.ema_indicator(dataframe['close'], window=20)
        dataframe['ema_50_1h']  = ta.trend.ema_indicator(dataframe['close'], window=50)
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

        dataframe['htf_ema20_50_dist'] = (
            (dataframe['ema_20_1h'] - dataframe['ema_50_1h']) / dataframe['ema_50_1h']
        )
        dataframe['htf_ema50_200_dist'] = (
            (dataframe['ema_50_1h'] - dataframe['ema_200_1h']) / dataframe['ema_200_1h']
        )
        dataframe['htf_price_ema200_dist'] = (
            (dataframe['close'] - dataframe['ema_200_1h']) / dataframe['ema_200_1h']
        )
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
        vol_mean = dataframe['volume'].rolling(50).mean()
        vol_std  = dataframe['volume'].rolling(50).std()
        dataframe['volume_zscore'] = (dataframe['volume'] - vol_mean) / vol_std

        dataframe = self._detect_swing_points(dataframe)
        dataframe = self._detect_order_blocks(dataframe)
        dataframe = self._detect_fvg(dataframe)
        dataframe = self._detect_liquidity(dataframe)
        dataframe = self._detect_kill_zone(dataframe)
        dataframe = self._detect_bos(dataframe)
        dataframe = self._compute_smc_features(dataframe)
        dataframe = self._compute_ml_score(dataframe)

        return dataframe

    def _detect_swing_points(self, df: DataFrame, lookback: int = 11) -> DataFrame:
        df['swing_high'] = (
            df['high'] == df['high'].rolling(lookback).max()
        ).astype(int)
        df['swing_low'] = (
            df['low'] == df['low'].rolling(lookback).min()
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

    def _detect_order_blocks(self, df: DataFrame) -> DataFrame:
        body     = abs(df['close'] - df['open'])
        avg_body = body.rolling(20).mean()
        impulse  = body > avg_body * 1.5

        df['ob_bullish'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            impulse &
            (df['close'] > df['open'])
        ).astype(int)
        df['ob_bull_top']    = df['open'].shift(1).where(df['ob_bullish'] == 1).ffill()
        df['ob_bull_bottom'] = df['close'].shift(1).where(df['ob_bullish'] == 1).ffill()

        df['ob_bearish'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            impulse &
            (df['close'] < df['open'])
        ).astype(int)
        df['ob_bear_top']    = df['close'].shift(1).where(df['ob_bearish'] == 1).ffill()
        df['ob_bear_bottom'] = df['open'].shift(1).where(df['ob_bearish'] == 1).ffill()
        return df

    def _detect_fvg(self, df: DataFrame) -> DataFrame:
        df['fvg_bullish'] = (
            df['high'].shift(2) < df['low']
        ).astype(int)
        df['fvg_bull_bottom'] = df['high'].shift(2).where(df['fvg_bullish'] == 1).ffill()
        df['fvg_bull_top']    = df['low'].where(df['fvg_bullish'] == 1).ffill()
        df['fvg_bearish'] = (
            df['low'].shift(2) > df['high']
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

    def _compute_smc_features(self, df: DataFrame) -> DataFrame:
        df['dist_to_ob'] = (
            (df['close'] - df['ob_bull_top']) / df['close']
        ).fillna(0)
        df['ob_size'] = (
            (df['ob_bull_top'] - df['ob_bull_bottom']) / df['close']
        ).fillna(0)

        ob_age, last_ob = [], np.nan
        for i in range(len(df)):
            if df['ob_bullish'].iloc[i] == 1:
                last_ob = i
            ob_age.append(i - last_ob if not np.isnan(last_ob) else 999)
        df['ob_age'] = ob_age

        df['dist_to_fvg'] = (
            (df['close'] - df['fvg_bull_top']) / df['close']
        ).fillna(0)
        df['fvg_size'] = (
            (df['fvg_bull_top'] - df['fvg_bull_bottom']) / df['close']
        ).fillna(0)
        df['dist_to_swing_high'] = (
            (df['last_swing_high'] - df['close']) / df['close']
        ).fillna(0)
        df['dist_to_swing_low'] = (
            (df['close'] - df['last_swing_low']) / df['close']
        ).fillna(0)
        df['atr_norm']  = df['atr'] / df['close']

        # ✅ Nombre correcto de producción: atr_1h_1h
        df['atr_ratio'] = df['atr'] / df['atr_1h_1h'].replace(0, np.nan).ffill()

        return df

    def _compute_ml_score(self, df: DataFrame) -> DataFrame:
        df['ml_score'] = 0.5
        if self.ml_model is None:
            return df
        features = df[self.FEATURE_COLS].fillna(0)
        try:
            scores = self.ml_model.predict_proba(features)[:, 1]
            df['ml_score'] = scores
        except Exception as e:
            print(f"⚠️  Error en ML score: {e}")
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ml_filter = (
            (dataframe['ml_score'] >= self.ml_score_threshold)
            if self.ml_model is not None
            else True
        )
        dataframe.loc[
            (
                (dataframe['htf_bias_1h'] == 1) &
                (dataframe['ema_9'] > dataframe['ema_21']) &
                (dataframe['rsi'] > 25) &
                (dataframe['rsi'] < 75) &
                (
                    (dataframe['close'] <= dataframe['ob_bull_top']) |
                    (dataframe['close'] <= dataframe['fvg_bull_top']) |
                    (dataframe['close'] <= dataframe['ema_21'] * 1.005)
                ) &
                (dataframe['volume_ratio'] > 0.8) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['volume'] > 0) &
                ml_filter
            ),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['bos_bearish'] == 1) |
                (dataframe['htf_bias_1h'] == -1)
            ),
            'exit_long'
        ] = 1
        return dataframe