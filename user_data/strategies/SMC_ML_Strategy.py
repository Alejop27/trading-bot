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

    # ── Umbral ML: solo entra si score >= este valor ─────────────────────────
    ml_score_threshold = 0.60
    ml_model = None
    ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'ml_models', 'smc_ml_model.pkl')

    def bot_start(self, **kwargs):
        """Carga el modelo ML al arrancar. Si no existe, opera sin filtro."""
        if os.path.exists(self.ml_model_path):
            self.ml_model = joblib.load(self.ml_model_path)
            print(f"✅ Modelo ML cargado: {self.ml_model_path}")
        else:
            print("⚠️  Modelo ML no encontrado. Operando sin filtro ML.")

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

        # ── Features continuas del HTF (tu Capa 1) ───────────────────────────
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

        # ── Z-score de volumen (tu sugerencia) ───────────────────────────────
        vol_std = dataframe['volume'].rolling(50).std()
        vol_mean = dataframe['volume'].rolling(50).mean()
        dataframe['volume_zscore'] = (dataframe['volume'] - vol_mean) / vol_std

        dataframe = self._detect_swing_points(dataframe)
        dataframe = self._detect_order_blocks(dataframe)
        dataframe = self._detect_fvg(dataframe)
        dataframe = self._detect_liquidity(dataframe)
        dataframe = self._detect_kill_zone(dataframe)
        dataframe = self._detect_bos(dataframe)

        # ── Features continuas SMC (tu Capa 2) ───────────────────────────────
        dataframe = self._compute_smc_features(dataframe)

        # ── Score ML ─────────────────────────────────────────────────────────
        dataframe = self._compute_ml_score(dataframe)

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

    def _compute_smc_features(self, df: DataFrame) -> DataFrame:
        """Features continuas SMC — tu Capa 2 y Capa 3."""

        # Distancia al Order Block (normalizada)
        df['dist_to_ob'] = (
            (df['close'] - df['ob_bull_top']) / df['close']
        ).fillna(0)

        # Tamaño del Order Block
        df['ob_size'] = (
            (df['ob_bull_top'] - df['ob_bull_bottom']) / df['close']
        ).fillna(0)

        # Antigüedad del OB: velas desde el último OB
        ob_indices = df.index[df['ob_bullish'] == 1].tolist()
        df['ob_age'] = 999
        for idx in df.index:
            past_obs = [i for i in ob_indices if i <= idx]
            if past_obs:
                df.loc[idx, 'ob_age'] = idx - past_obs[-1]

        # Distancia al FVG
        df['dist_to_fvg'] = (
            (df['close'] - df['fvg_bull_top']) / df['close']
        ).fillna(0)

        # Tamaño del FVG
        df['fvg_size'] = (
            (df['fvg_bull_top'] - df['fvg_bull_bottom']) / df['close']
        ).fillna(0)

        # Distancia a swing highs/lows (liquidez)
        df['dist_to_swing_high'] = (
            (df['last_swing_high'] - df['close']) / df['close']
        ).fillna(0)
        df['dist_to_swing_low'] = (
            (df['close'] - df['last_swing_low']) / df['close']
        ).fillna(0)

        # ATR normalizado
        df['atr_norm'] = df['atr'] / df['close']

        # Ratio ATR 15m vs 1h
        df['atr_ratio'] = df['atr'] / df['atr_1h'].replace(0, np.nan).ffill()

        return df

    def _compute_ml_score(self, df: DataFrame) -> DataFrame:
        """Aplica el modelo ML y genera score de probabilidad de ROI."""
        df['ml_score'] = 0.5  # default neutral si no hay modelo

        if self.ml_model is None:
            return df

        feature_cols = self._get_feature_cols()
        features = df[feature_cols].fillna(0)

        try:
            scores = self.ml_model.predict_proba(features)[:, 1]
            df['ml_score'] = scores
        except Exception as e:
            print(f"⚠️  Error en ML score: {e}")

        return df

    def _get_feature_cols(self) -> list:
        """Lista canónica de features para el modelo ML."""
        return [
            # Capa 1: Régimen HTF
            'htf_bias_1h',
            'htf_ema20_50_dist_1h',
            'htf_ema50_200_dist_1h',
            'htf_price_ema200_dist_1h',
            # Capa 2: Estructura SMC continua
            'dist_to_ob',
            'ob_size',
            'ob_age',
            'dist_to_fvg',
            'fvg_size',
            # Capa 3: Liquidez
            'dist_to_swing_high',
            'dist_to_swing_low',
            # Capa 4: Volumen
            'volume_ratio',
            'volume_zscore',
            # Capa 5: Volatilidad
            'atr_norm',
            'atr_ratio',
            'bb_width',
            # Capa 6: Momento
            'rsi',
            'macd_hist',
            # Contexto
            'in_kill_zone',
            'bos_bullish',
        ]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Condición ML: si hay modelo usa el score, si no omite el filtro
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