# analyze_scores.py
# Valida si el score ML es monotónico con el win rate real

import pandas as pd
import numpy as np
import json
import zipfile
import glob
import joblib
from pathlib import Path
import ta

USERDATA_PATH = Path("user_data")
BACKTEST_PATH = USERDATA_PATH / "backtest_results"
DATA_PATH     = USERDATA_PATH / "data" / "bybit"
MODEL_PATH    = USERDATA_PATH / "ml_models" / "smc_ml_model.pkl"
SCALER_PATH   = USERDATA_PATH / "ml_models" / "smc_ml_scaler.pkl"
PAIR          = "BTC_USDT"

FEATURE_COLS = [
    'htf_bias', 'htf_ema20_50_dist', 'htf_ema50_200_dist',
    'htf_price_ema200_dist', 'dist_to_ob', 'ob_size', 'ob_age',
    'dist_to_fvg', 'fvg_size', 'dist_to_swing_high', 'dist_to_swing_low',
    'volume_ratio', 'volume_zscore', 'atr_norm', 'atr_ratio',
    'bb_width', 'rsi', 'macd_hist', 'in_kill_zone', 'bos_bullish',
]

def load_trades():
    files = sorted(glob.glob(str(BACKTEST_PATH / "*.zip")))
    with zipfile.ZipFile(files[-1]) as z:
        json_files = [f for f in z.namelist() if f.endswith(".json")]
        with z.open(json_files[0]) as f:
            data = json.load(f)
    key    = list(data["strategy"].keys())[0]
    trades = pd.DataFrame(data["strategy"][key]["trades"])
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["target"]    = (trades["exit_reason"] == "roi").astype(int)
    return trades

def load_features():
    df_15m = pd.read_feather(DATA_PATH / f"{PAIR}-15m.feather")
    df_1h  = pd.read_feather(DATA_PATH / f"{PAIR}-1h.feather")
    df_15m["date"] = pd.to_datetime(df_15m["date"], unit="ms", utc=True)
    df_1h["date"]  = pd.to_datetime(df_1h["date"],  unit="ms", utc=True)

    # Indicadores 15m
    df_15m["rsi"]       = ta.momentum.rsi(df_15m["close"], window=14)
    macd                = ta.trend.MACD(df_15m["close"])
    df_15m["macd_hist"] = macd.macd_diff()
    df_15m["atr"]       = ta.volatility.average_true_range(
        df_15m["high"], df_15m["low"], df_15m["close"], window=14)
    bb = ta.volatility.BollingerBands(df_15m["close"], window=20, window_dev=2)
    df_15m["bb_width"]      = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df_15m["volume_ma_20"]  = df_15m["volume"].rolling(20).mean()
    df_15m["volume_ratio"]  = df_15m["volume"] / df_15m["volume_ma_20"]
    df_15m["volume_zscore"] = (
        (df_15m["volume"] - df_15m["volume"].rolling(50).mean()) /
        df_15m["volume"].rolling(50).std()
    )

    # Swing points
    df_15m["swing_high"]      = (df_15m["high"] == df_15m["high"].rolling(11, center=True).max()).astype(int)
    df_15m["swing_low"]       = (df_15m["low"]  == df_15m["low"].rolling(11,  center=True).min()).astype(int)
    df_15m["last_swing_high"] = df_15m["high"].where(df_15m["swing_high"] == 1).ffill()
    df_15m["last_swing_low"]  = df_15m["low"].where(df_15m["swing_low"]  == 1).ffill()

    # BOS
    df_15m["bos_bullish"] = (
        (df_15m["close"] > df_15m["last_swing_high"].shift(1)) &
        (df_15m["close"].shift(1) <= df_15m["last_swing_high"].shift(1))
    ).astype(int)

    # Order Blocks
    body     = abs(df_15m["close"] - df_15m["open"])
    avg_body = body.rolling(20).mean()
    impulse  = body > avg_body * 1.5
    df_15m["ob_bullish"]     = (
        (df_15m["close"] < df_15m["open"]) & impulse.shift(-1) &
        (df_15m["close"].shift(-1) > df_15m["open"].shift(-1))
    ).astype(int)
    df_15m["ob_bull_top"]    = df_15m["open"].where(df_15m["ob_bullish"] == 1).ffill()
    df_15m["ob_bull_bottom"] = df_15m["close"].where(df_15m["ob_bullish"] == 1).ffill()

    # FVG
    df_15m["fvg_bullish"]     = (df_15m["high"].shift(1) < df_15m["low"].shift(-1)).astype(int)
    df_15m["fvg_bull_top"]    = df_15m["low"].shift(-1).where(df_15m["fvg_bullish"] == 1).ffill()
    df_15m["fvg_bull_bottom"] = df_15m["high"].shift(1).where(df_15m["fvg_bullish"] == 1).ffill()

    # Kill zones
    hour = df_15m["date"].dt.hour
    df_15m["in_kill_zone"] = (((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15))).astype(int)

    # Features continuas
    df_15m["dist_to_ob"]         = ((df_15m["close"] - df_15m["ob_bull_top"]) / df_15m["close"]).fillna(0)
    df_15m["ob_size"]            = ((df_15m["ob_bull_top"] - df_15m["ob_bull_bottom"]) / df_15m["close"]).fillna(0)
    df_15m["dist_to_fvg"]        = ((df_15m["close"] - df_15m["fvg_bull_top"]) / df_15m["close"]).fillna(0)
    df_15m["fvg_size"]           = ((df_15m["fvg_bull_top"] - df_15m["fvg_bull_bottom"]) / df_15m["close"]).fillna(0)
    df_15m["dist_to_swing_high"] = ((df_15m["last_swing_high"] - df_15m["close"]) / df_15m["close"]).fillna(0)
    df_15m["dist_to_swing_low"]  = ((df_15m["close"] - df_15m["last_swing_low"]) / df_15m["close"]).fillna(0)
    df_15m["atr_norm"]           = df_15m["atr"] / df_15m["close"]

    ob_age_list, last_ob = [], np.nan
    for i in range(len(df_15m)):
        if df_15m["ob_bullish"].iloc[i] == 1:
            last_ob = i
        ob_age_list.append(i - last_ob if not np.isnan(last_ob) else 999)
    df_15m["ob_age"] = ob_age_list

    # HTF merge
    df_1h["ema_20_1h"]  = ta.trend.ema_indicator(df_1h["close"], window=20)
    df_1h["ema_50_1h"]  = ta.trend.ema_indicator(df_1h["close"], window=50)
    df_1h["ema_200_1h"] = ta.trend.ema_indicator(df_1h["close"], window=200)
    df_1h["atr_1h"]     = ta.volatility.average_true_range(
        df_1h["high"], df_1h["low"], df_1h["close"], window=14)
    df_1h["htf_bias"] = 0
    df_1h.loc[(df_1h["ema_20_1h"] > df_1h["ema_50_1h"]) & (df_1h["ema_50_1h"] > df_1h["ema_200_1h"]), "htf_bias"] = 1
    df_1h.loc[(df_1h["ema_20_1h"] < df_1h["ema_50_1h"]) & (df_1h["ema_50_1h"] < df_1h["ema_200_1h"]), "htf_bias"] = -1
    df_1h["htf_ema20_50_dist"]     = (df_1h["ema_20_1h"] - df_1h["ema_50_1h"]) / df_1h["ema_50_1h"]
    df_1h["htf_ema50_200_dist"]    = (df_1h["ema_50_1h"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]
    df_1h["htf_price_ema200_dist"] = (df_1h["close"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]

    df_1h_feat = df_1h[["date", "atr_1h", "htf_bias",
                          "htf_ema20_50_dist", "htf_ema50_200_dist",
                          "htf_price_ema200_dist"]].copy()
    df_15m = pd.merge_asof(df_15m.sort_values("date"),
                            df_1h_feat.sort_values("date"),
                            on="date", direction="backward")
    df_15m["atr_ratio"] = df_15m["atr"] / df_15m["atr_1h"].replace(0, np.nan)

    return df_15m.dropna(subset=["rsi", "atr", "htf_bias"]).set_index("date")

if __name__ == "__main__":
    print("Cargando modelo y datos...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    trades = load_trades()
    feats  = load_features()

    # Asignar score ML a cada trade
    rows = []
    for _, trade in trades.iterrows():
        available = feats.index[feats.index <= trade["open_date"]]
        if len(available) == 0:
            continue
        candle = feats.loc[available[-1]]
        X      = pd.DataFrame([{col: candle.get(col, 0) for col in FEATURE_COLS}])
        X_sc   = scaler.transform(X.fillna(0))
        score  = model.predict_proba(X_sc)[0, 1]
        rows.append({
            "open_date":   trade["open_date"],
            "exit_reason": trade["exit_reason"],
            "target":      trade["target"],
            "ml_score":    score
        })

    df = pd.DataFrame(rows)

    # Distribución de scores por bins
    df["score_bin"] = pd.cut(df["ml_score"],
                              bins=[0, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0],
                              labels=["<0.50", "0.50-0.55", "0.55-0.60",
                                      "0.60-0.65", "0.65-0.70", ">0.70"])

    print("\n📊 Win Rate por Score ML:")
    print("-" * 55)
    summary = df.groupby("score_bin", observed=True).agg(
        trades   = ("target", "count"),
        wins     = ("target", "sum"),
        win_rate = ("target", "mean")
    ).round(3)
    summary["win_rate_pct"] = (summary["win_rate"] * 100).round(1)
    print(summary[["trades", "wins", "win_rate_pct"]].to_string())

    print("\n📊 Exit reason por score alto (>0.65):")
    high_score = df[df["ml_score"] > 0.65]
    print(high_score["exit_reason"].value_counts().to_string())

    print("\n📊 Exit reason por score bajo (<0.50):")
    low_score = df[df["ml_score"] < 0.50]
    print(low_score["exit_reason"].value_counts().to_string())

    print(f"\n📈 Score promedio trades ROI:     {df[df['target']==1]['ml_score'].mean():.3f}")
    print(f"📉 Score promedio trades No-ROI:  {df[df['target']==0]['ml_score'].mean():.3f}")
    print(f"\n✅ ¿Hay monotonicidad? Revisa si win_rate_pct sube con el score.")