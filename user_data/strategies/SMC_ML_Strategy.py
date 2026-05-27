# ─────────────────────────────────────────────────────────────────────────────
# SMC_ML_Strategy.py
# Estrategia híbrida: Smart Money Concepts + Filtro ML
# Exchange: Binance | Par: BTC/USDT | Timeframe: 15m
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

    # ── Configuración base ──────────────────────────────────────────────────
    INTERFACE_VERSION = 3
    timeframe = '15m'
    can_short = False  # solo long por ahora

    # Gestión de riesgo — estándar institucional
    minimal_roi = {
        "0":  0.03,    # 3% TP inmediato si se da
        "30": 0.02,    # 2% después de 30 min
        "60": 0.01,    # 1% después de 1h
        "120": 0       # breakeven después de 2h
    }
    stoploss = -0.02           # 2% stop loss máximo
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    # Freqtrade settings
    startup_candle_count = 200  # velas necesarias para calcular indicadores
    process_only_new_candles = True

    # ── Timeframe superior para bias direccional ─────────────────────────────
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Tendencia en 1h
        dataframe['ema_20_1h'] = ta.trend.ema_indicator(dataframe['close'], window=20)
        dataframe['ema_50_1h'] = ta.trend.ema_indicator(dataframe['close'], window=50)
        dataframe['ema_200_1h'] = ta.trend.ema_indicator(dataframe['close'], window=200)

        # Bias: 1 = alcista, -1 = bajista, 0 = neutral
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

        # ATR para contexto de volatilidad
        dataframe['atr_1h'] = ta.volatility.average_true_range(
            dataframe['high'], dataframe['low'], dataframe['close'], window=14
        )
        return dataframe

    # ── Indicadores principales (15m) ────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # --- Tendencia 15m ---
        dataframe['ema_9']   = ta.trend.ema_indicator(dataframe['close'], window=9)
        dataframe['ema_21']  = ta.trend.ema_indicator(dataframe['close'], window=21)
        dataframe['ema_50']  = ta.trend.ema_indicator(dataframe['close'], window=50)
        dataframe['ema_200'] = ta.trend.ema_indicator(dataframe['close'], window=200)

        # --- Momentum ---
        dataframe['rsi'] = ta.momentum.rsi(dataframe['close'], window=14)
        macd = ta.trend.MACD(dataframe['close'])
        dataframe['macd']        = macd.macd()
        dataframe['macd_signal'] = macd.macd_signal()
        dataframe['macd_hist']   = macd.macd_diff()

        # --- Volatilidad ---
        dataframe['atr'] = ta.volatility.average_true_range(
            dataframe['high'], dataframe['low'], dataframe['close'], window=14
        )
        bb = ta.volatility.BollingerBands(dataframe['close'], window=20, window_dev=2)
        dataframe['bb_upper']  = bb.bollinger_hband()
        dataframe['bb_lower']  = bb.bollinger_lband()
        dataframe['bb_mid']    = bb.bollinger_mavg()
        dataframe['bb_width']  = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']

        # --- Volumen ---
        dataframe['volume_ma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma_20']

        # ── SMC: Estructura de mercado ────────────────────────────────────────
        dataframe = self._detect_swing_points(dataframe)
        dataframe = self._detect_order_blocks(dataframe)
        dataframe = self._detect_fvg(dataframe)
        dataframe = self._detect_liquidity(dataframe)
        dataframe = self._detect_kill_zone(dataframe)
        dataframe = self._detect_bos(dataframe)

        return dataframe

    # ── SMC: Swing Points ────────────────────────────────────────────────────
    def _detect_swing_points(self, df: DataFrame, lookback: int = 5) -> DataFrame:
        df['swing_high'] = (
            (df['high'] == df['high'].rolling(lookback * 2 + 1, center=True).max())
        ).astype(int)
        df['swing_low'] = (
            (df['low'] == df['low'].rolling(lookback * 2 + 1, center=True).min())
        ).astype(int)

        # Guardar niveles de swing para referencia
        df['last_swing_high'] = df['high'].where(df['swing_high'] == 1).ffill()
        df['last_swing_low']  = df['low'].where(df['swing_low'] == 1).ffill()
        return df

    # ── SMC: Break of Structure ──────────────────────────────────────────────
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

    # ── SMC: Order Blocks ────────────────────────────────────────────────────
    def _detect_order_blocks(self, df: DataFrame, lookback: int = 3) -> DataFrame:
        body = abs(df['close'] - df['open'])
        avg_body = body.rolling(20).mean()
        impulse = body > avg_body * 1.5

        # Bullish OB: vela bajista antes de impulso alcista
        df['ob_bullish'] = (
            (df['close'] < df['open']) &           # vela bajista
            impulse.shift(-1) &                     # impulso en siguiente vela
            (df['close'].shift(-1) > df['open'].shift(-1))  # siguiente es alcista
        ).astype(int)

        df['ob_bull_top']    = df['open'].where(df['ob_bullish'] == 1).ffill()
        df['ob_bull_bottom'] = df['close'].where(df['ob_bullish'] == 1).ffill()

        # Bearish OB: vela alcista antes de impulso bajista
        df['ob_bearish'] = (
            (df['close'] > df['open']) &
            impulse.shift(-1) &
            (df['close'].shift(-1) < df['open'].shift(-1))
        ).astype(int)

        df['ob_bear_top']    = df['close'].where(df['ob_bearish'] == 1).ffill()
        df['ob_bear_bottom'] = df['open'].where(df['ob_bearish'] == 1).ffill()

        return df

    # ── SMC: Fair Value Gaps ─────────────────────────────────────────────────
    def _detect_fvg(self, df: DataFrame) -> DataFrame:
        # Bullish FVG: gap entre high de vela -1 y low de vela +1
        df['fvg_bullish'] = (
            df['high'].shift(1) < df['low'].shift(-1)
        ).astype(int)

        df['fvg_bull_bottom'] = df['high'].shift(1).where(df['fvg_bullish'] == 1).ffill()
        df['fvg_bull_top']    = df['low'].shift(-1).where(df['fvg_bullish'] == 1).ffill()

        # Bearish FVG
        df['fvg_bearish'] = (
            df['low'].shift(1) > df['high'].shift(-1)
        ).astype(int)
        return df

    # ── SMC: Zonas de Liquidez ───────────────────────────────────────────────
    def _detect_liquidity(self, df: DataFrame, tolerance: float = 0.001) -> DataFrame:
        # Equal highs/lows en ventana de 20 velas
        for i in range(20, len(df)):
            window_highs = df['high'].iloc[i-20:i]
            curr_high = df['high'].iloc[i]
            equal_highs = (abs(window_highs - curr_high) / curr_high < tolerance).sum()
            df.iloc[i, df.columns.get_loc('liquidity_high') if 'liquidity_high' in df.columns
                    else df.columns.get_loc(df.columns[-1])] = (equal_highs >= 2)

        df['liquidity_high'] = (
            (df['high'].rolling(20).max() - df['high']).abs() / df['high'] < tolerance
        ).astype(int)

        df['liquidity_low'] = (
            (df['low'].rolling(20).min() - df['low']).abs() / df['low'] < tolerance
        ).astype(int)
        return df

    # ── SMC: Kill Zones (UTC) ────────────────────────────────────────────────
    def _detect_kill_zone(self, df: DataFrame) -> DataFrame:
        hour = df['date'].dt.hour if hasattr(df['date'], 'dt') else pd.to_datetime(df['date']).dt.hour

        # London Open: 07:00-10:00 UTC
        # NY Open: 12:00-15:00 UTC
        df['in_kill_zone'] = (
            ((hour >= 7) & (hour < 10)) |   # London Open
            ((hour >= 12) & (hour < 15))     # NY Open
        ).astype(int)

        df['london_open'] = ((hour >= 7) & (hour < 10)).astype(int)
        df['ny_open']     = ((hour >= 12) & (hour < 15)).astype(int)
        return df

    # ── Señal de entrada ─────────────────────────────────────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 1. Bias HTF alcista
                (dataframe['htf_bias_1h'] == 1) &

                # 2. Break of Structure alcista confirmado
                (dataframe['bos_bullish'] == 1) &

                # 3. Precio retrocede a Order Block o FVG
                (
                    (dataframe['close'] <= dataframe['ob_bull_top']) |
                    (dataframe['close'] <= dataframe['fvg_bull_top'])
                ) &

                # 4. RSI no sobrecomprado
                (dataframe['rsi'] < 70) &
                (dataframe['rsi'] > 30) &

                # 5. En Kill Zone
                (dataframe['in_kill_zone'] == 1) &

                # 6. Volumen confirmando
                (dataframe['volume_ratio'] > 1.2) &

                # 7. Vela de confirmación alcista
                (dataframe['close'] > dataframe['open']) &

                # 8. Volumen presente
                (dataframe['volume'] > 0)
            ),
            'enter_long'
        ] = 1
        return dataframe

    # ── Señal de salida ──────────────────────────────────────────────────────
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Salida: BOS bajista o RSI sobrecomprado o bias cambia
                (dataframe['bos_bearish'] == 1) |
                (dataframe['rsi'] > 80) |
                (dataframe['htf_bias_1h'] == -1)
            ),
            'exit_long'
        ] = 1
        return dataframe