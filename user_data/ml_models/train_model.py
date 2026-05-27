# ─────────────────────────────────────────────────────────────────────────────
# train_model.py
# Entrena el filtro ML con los trades históricos del backtest
# Ejecutar desde: C:\trading-bot\
# Uso: python user_data/ml_models/train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import glob
import joblib
import os
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# ── Configuración ─────────────────────────────────────────────────────────────
USERDATA_PATH   = Path("user_data")
BACKTEST_PATH   = USERDATA_PATH / "backtest_results"
DATA_PATH       = USERDATA_PATH / "data" / "bybit"
MODEL_PATH      = USERDATA_PATH / "ml_models" / "smc_ml_model.pkl"
SCALER_PATH     = USERDATA_PATH / "ml_models" / "smc_ml_scaler.pkl"
TIMEFRAME       = "15m"
PAIR            = "BTC_USDT"

FEATURE_COLS = [
    'htf_bias',
    'htf_ema20_50_dist',
    'htf_ema50_200_dist',
    'htf_price_ema200_dist',
    'dist_to_ob',
    'ob_size',
    'ob_age',
    'dist_to_fvg',
    'fvg_size',
    'dist_to_swing_high',
    'dist_to_swing_low',
    'volume_ratio',
    'volume_zscore',
    'atr_norm',
    'atr_ratio',
    'bb_width',
    'rsi',
    'macd_hist',
    'in_kill_zone',
    'bos_bullish',
]

# ── 1. Cargar trades del backtest ─────────────────────────────────────────────

def load_backtest_trades() -> pd.DataFrame:
    """Carga el backtest más reciente exportado."""
    files = sorted(glob.glob(str(BACKTEST_PATH / "*.json")))
    # Filtra archivos meta y carga el más reciente
    trade_files = [f for f in files if "meta" not in f]

    if not trade_files:
        raise FileNotFoundError(
            f"No se encontraron archivos de backtest en {BACKTEST_PATH}\n"
            "Asegúrate de haber corrido: freqtrade backtesting --export trades"
        )

    latest = trade_files[-1]
    print(f"📂 Cargando trades desde: {latest}")

    with open(latest, "r") as f:
        data = json.load(f)

    # Extraer trades de la estrategia
    strategy_key = list(data["strategy"].keys())[0]
    trades = data["strategy"][strategy_key]["trades"]
    df = pd.DataFrame(trades)

    print(f"✅ Trades cargados: {len(df)}")
    print(f"   Columnas disponibles: {list(df.columns)}")
    return df


# ── 2. Cargar datos OHLCV ─────────────────────────────────────────────────────

def load_ohlcv() -> pd.DataFrame:
    """Carga datos históricos de BTC/USDT 15m."""
    file_15m = DATA_PATH / f"{PAIR}-{TIMEFRAME}.feather"
    file_1h  = DATA_PATH / f"{PAIR}-1h.feather"

    if not file_15m.exists():
        # Intenta formato JSON
        file_15m = DATA_PATH / f"{PAIR}-{TIMEFRAME}.json.gz"
        file_1h  = DATA_PATH / f"{PAIR}-1h.json.gz"

    print(f"📂 Cargando OHLCV 15m desde: {file_15m}")

    if str(file_15m).endswith(".feather"):
        df_15m = pd.read_feather(file_15m)
        df_1h  = pd.read_feather(file_1h)
    else:
        df_15m = pd.read_json(file_15m)
        df_1h  = pd.read_json(file_1h)
        df_15m.columns = ["date", "open", "high", "low", "close", "volume"]
        df_1h.columns  = ["date", "open", "high", "low", "close", "volume"]

    df_15m["date"] = pd.to_datetime(df_15m["date"], unit="ms", utc=True)
    df_1h["date"]  = pd.to_datetime(df_1h["date"],  unit="ms", utc=True)

    df_15m = df_15m.sort_values("date").reset_index(drop=True)
    df_1h  = df_1h.sort_values("date").reset_index(drop=True)

    print(f"✅ OHLCV 15m: {len(df_15m)} velas | 1h: {len(df_1h)} velas")
    return df_15m, df_1h


# ── 3. Calcular features ──────────────────────────────────────────────────────

def compute_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Recalcula todas las features SMC sobre el histórico completo."""
    import ta as ta_lib

    df = df_15m.copy()

    # ── Indicadores 15m ───────────────────────────────────────────────────────
    df["ema_9"]   = ta_lib.trend.ema_indicator(df["close"], window=9)
    df["ema_21"]  = ta_lib.trend.ema_indicator(df["close"], window=21)
    df["ema_50"]  = ta_lib.trend.ema_indicator(df["close"], window=50)
    df["ema_200"] = ta_lib.trend.ema_indicator(df["close"], window=200)

    df["rsi"] = ta_lib.momentum.rsi(df["close"], window=14)

    macd = ta_lib.trend.MACD(df["close"])
    df["macd_hist"] = macd.macd_diff()

    df["atr"] = ta_lib.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )
    bb = ta_lib.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

    df["volume_ma_20"]  = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_ma_20"]
    df["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(50).mean()) /
        df["volume"].rolling(50).std()
    )

    # ── Swing points ──────────────────────────────────────────────────────────
    df["swing_high"] = (
        df["high"] == df["high"].rolling(11, center=True).max()
    ).astype(int)
    df["swing_low"] = (
        df["low"] == df["low"].rolling(11, center=True).min()
    ).astype(int)
    df["last_swing_high"] = df["high"].where(df["swing_high"] == 1).ffill()
    df["last_swing_low"]  = df["low"].where(df["swing_low"] == 1).ffill()

    # ── BOS ───────────────────────────────────────────────────────────────────
    df["bos_bullish"] = (
        (df["close"] > df["last_swing_high"].shift(1)) &
        (df["close"].shift(1) <= df["last_swing_high"].shift(1))
    ).astype(int)

    # ── Order Blocks ──────────────────────────────────────────────────────────
    body     = abs(df["close"] - df["open"])
    avg_body = body.rolling(20).mean()
    impulse  = body > avg_body * 1.5

    df["ob_bullish"] = (
        (df["close"] < df["open"]) &
        impulse.shift(-1) &
        (df["close"].shift(-1) > df["open"].shift(-1))
    ).astype(int)
    df["ob_bull_top"]    = df["open"].where(df["ob_bullish"] == 1).ffill()
    df["ob_bull_bottom"] = df["close"].where(df["ob_bullish"] == 1).ffill()

    # ── FVG ───────────────────────────────────────────────────────────────────
    df["fvg_bullish"]    = (df["high"].shift(1) < df["low"].shift(-1)).astype(int)
    df["fvg_bull_top"]   = df["low"].shift(-1).where(df["fvg_bullish"] == 1).ffill()
    df["fvg_bull_bottom"] = df["high"].shift(1).where(df["fvg_bullish"] == 1).ffill()

    # ── Kill zones ────────────────────────────────────────────────────────────
    hour = df["date"].dt.hour
    df["in_kill_zone"] = (
        ((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15))
    ).astype(int)

    # ── Features continuas SMC ────────────────────────────────────────────────
    df["dist_to_ob"] = (
        (df["close"] - df["ob_bull_top"]) / df["close"]
    ).fillna(0)
    df["ob_size"] = (
        (df["ob_bull_top"] - df["ob_bull_bottom"]) / df["close"]
    ).fillna(0)

    # Antigüedad del OB (vectorizada)
    ob_mask = df["ob_bullish"] == 1
    ob_idx  = df.index[ob_mask]
    df["ob_age"] = np.nan
    last_ob = np.nan
    for i in df.index:
        if ob_mask[i]:
            last_ob = i
        df.at[i, "ob_age"] = (i - last_ob) if not np.isnan(last_ob) else 999

    df["dist_to_fvg"] = (
        (df["close"] - df["fvg_bull_top"]) / df["close"]
    ).fillna(0)
    df["fvg_size"] = (
        (df["fvg_bull_top"] - df["fvg_bull_bottom"]) / df["close"]
    ).fillna(0)
    df["dist_to_swing_high"] = (
        (df["last_swing_high"] - df["close"]) / df["close"]
    ).fillna(0)
    df["dist_to_swing_low"] = (
        (df["close"] - df["last_swing_low"]) / df["close"]
    ).fillna(0)
    df["atr_norm"] = df["atr"] / df["close"]

    # ── Features HTF (merge 1h → 15m) ────────────────────────────────────────
    df_1h = df_1h.copy()
    df_1h["ema_20_1h"]  = ta_lib.trend.ema_indicator(df_1h["close"], window=20)
    df_1h["ema_50_1h"]  = ta_lib.trend.ema_indicator(df_1h["close"], window=50)
    df_1h["ema_200_1h"] = ta_lib.trend.ema_indicator(df_1h["close"], window=200)
    df_1h["atr_1h"]     = ta_lib.volatility.average_true_range(
        df_1h["high"], df_1h["low"], df_1h["close"], window=14
    )
    df_1h["htf_bias"] = 0
    df_1h.loc[
        (df_1h["ema_20_1h"] > df_1h["ema_50_1h"]) &
        (df_1h["ema_50_1h"] > df_1h["ema_200_1h"]), "htf_bias"
    ] = 1
    df_1h.loc[
        (df_1h["ema_20_1h"] < df_1h["ema_50_1h"]) &
        (df_1h["ema_50_1h"] < df_1h["ema_200_1h"]), "htf_bias"
    ] = -1

    df_1h["htf_ema20_50_dist"]    = (df_1h["ema_20_1h"] - df_1h["ema_50_1h"]) / df_1h["ema_50_1h"]
    df_1h["htf_ema50_200_dist"]   = (df_1h["ema_50_1h"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]
    df_1h["htf_price_ema200_dist"] = (df_1h["close"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]

    # Merge: asof join (cada vela 15m toma el valor 1h más reciente)
    df_1h_feat = df_1h[["date", "atr_1h", "htf_bias",
                          "htf_ema20_50_dist", "htf_ema50_200_dist",
                          "htf_price_ema200_dist"]].copy()
    df = pd.merge_asof(
        df.sort_values("date"),
        df_1h_feat.sort_values("date"),
        on="date", direction="backward"
    )

    df["atr_ratio"] = df["atr"] / df["atr_1h"].replace(0, np.nan)

    return df.dropna(subset=["rsi", "atr", "htf_bias"])


# ── 4. Construir dataset de entrenamiento ─────────────────────────────────────

def build_training_set(trades: pd.DataFrame,
                       features_df: pd.DataFrame) -> tuple:
    """
    Para cada trade, extrae las features en la vela de entrada.
    Target: 1 si exit_reason == 'roi', 0 en caso contrario.
    """
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["target"]    = (trades["exit_reason"] == "roi").astype(int)

    features_df = features_df.set_index("date")

    rows = []
    for _, trade in trades.iterrows():
        entry_time = trade["open_date"]
        # Busca la vela de entrada más cercana (hacia atrás)
        available = features_df.index[features_df.index <= entry_time]
        if len(available) == 0:
            continue
        entry_candle = features_df.loc[available[-1]]

        row = {col: entry_candle.get(col, 0) for col in FEATURE_COLS}
        row["target"]     = trade["target"]
        row["open_date"]  = entry_time
        row["exit_reason"] = trade["exit_reason"]
        rows.append(row)

    df_train = pd.DataFrame(rows).sort_values("open_date")
    print(f"\n📊 Dataset construido: {len(df_train)} muestras")
    print(f"   ROI (1): {df_train['target'].sum()} | No-ROI (0): {(df_train['target']==0).sum()}")
    print(f"   Balance de clases: {df_train['target'].mean():.1%} positivos")

    X = df_train[FEATURE_COLS].fillna(0)
    y = df_train["target"]
    dates = df_train["open_date"]

    return X, y, dates


# ── 5. Entrenar modelo ────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, dates: pd.Series):
    """Walk-forward training con TimeSeriesSplit."""

    print("\n🤖 Iniciando entrenamiento con walk-forward validation...")

    tscv = TimeSeriesSplit(n_splits=3)
    scaler = RobustScaler()

    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = scaler.fit_transform(X.iloc[train_idx])
        X_test  = scaler.transform(X.iloc[test_idx])
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.60).astype(int)
        auc    = roc_auc_score(y_test, y_prob)
        fold_aucs.append(auc)

        print(f"\n  Fold {fold+1} — ROC-AUC: {auc:.3f}")
        print(classification_report(y_test, y_pred,
              target_names=["No-ROI", "ROI"], zero_division=0))

    print(f"\n📈 AUC promedio walk-forward: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # ── Entrenamiento final con todos los datos ───────────────────────────────
    print("\n🔧 Entrenando modelo final con todos los datos...")
    X_scaled = scaler.fit_transform(X)

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum() / (y==1).sum(),
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    final_model.fit(X_scaled, y)

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.Series(
        final_model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print("\n📊 Feature Importance (top 10):")
    print(importance.head(10).to_string())

    # ── Guardar modelo y scaler ───────────────────────────────────────────────
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Modelo guardado: {MODEL_PATH}")
    print(f"✅ Scaler guardado: {SCALER_PATH}")

    return final_model, scaler, importance


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SMC ML Model Trainer")
    print("=" * 60)

    trades      = load_backtest_trades()
    df_15m, df_1h = load_ohlcv()
    features_df = compute_features(df_15m, df_1h)
    X, y, dates = build_training_set(trades, features_df)

    if len(X) < 30:
        print(f"\n⚠️  Solo {len(X)} trades disponibles.")
        print("   Se recomienda mínimo 100 trades para un modelo confiable.")
        print("   Considera ampliar el rango de datos del backtest.")
    else:
        model, scaler, importance = train_model(X, y, dates)
        print("\n🎯 Entrenamiento completado.")
        print("   El modelo está listo para usarse como filtro en la estrategia.")