# ─────────────────────────────────────────────────────────────────────────────
# train_model.py — Anti-Leakage v2
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json
import glob
import joblib
import zipfile
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

USERDATA_PATH = Path("user_data")
BACKTEST_PATH = USERDATA_PATH / "backtest_results"
DATA_PATH     = USERDATA_PATH / "data" / "bybit"
MODEL_PATH    = USERDATA_PATH / "ml_models" / "smc_ml_model.pkl"
SCALER_PATH   = USERDATA_PATH / "ml_models" / "smc_ml_scaler.pkl"
PAIR          = "BTC_USDT"

FEATURE_COLS = [
    'htf_bias', 'htf_ema20_50_dist', 'htf_ema50_200_dist',
    'htf_price_ema200_dist',
    'dist_to_ob', 'ob_size', 'ob_age',
    'dist_to_fvg', 'fvg_size',
    'dist_to_swing_high', 'dist_to_swing_low',
    'volume_ratio', 'volume_zscore',
    'atr_norm', 'atr_ratio', 'bb_width',
    'rsi', 'macd_hist',
    'in_kill_zone', 'bos_bullish',
]

def load_backtest_trades() -> pd.DataFrame:
    files = sorted(glob.glob(str(BACKTEST_PATH / "*.zip")))
    if not files:
        raise FileNotFoundError("No se encontraron archivos de backtest.")
    with zipfile.ZipFile(files[-1]) as z:
        json_files = [f for f in z.namelist() if f.endswith(".json")]
        with z.open(json_files[0]) as f:
            data = json.load(f)
    key    = list(data["strategy"].keys())[0]
    trades = pd.DataFrame(data["strategy"][key]["trades"])
    print(f"✅ Trades: {len(trades)} | {trades['exit_reason'].value_counts().to_dict()}")
    return trades

def load_ohlcv():
    df_15m = pd.read_feather(DATA_PATH / f"{PAIR}-15m.feather")
    df_1h  = pd.read_feather(DATA_PATH / f"{PAIR}-1h.feather")
    df_15m["date"] = pd.to_datetime(df_15m["date"], unit="ms", utc=True)
    df_1h["date"]  = pd.to_datetime(df_1h["date"],  unit="ms", utc=True)
    df_15m = df_15m.sort_values("date").reset_index(drop=True)
    df_1h  = df_1h.sort_values("date").reset_index(drop=True)
    print(f"✅ OHLCV 15m: {len(df_15m)} | 1h: {len(df_1h)}")
    return df_15m, df_1h

def compute_features(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    import ta as ta_lib

    df = df_15m.copy()

    df["rsi"]       = ta_lib.momentum.rsi(df["close"], window=14)
    macd            = ta_lib.trend.MACD(df["close"])
    df["macd_hist"] = macd.macd_diff()
    df["atr"]       = ta_lib.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14)
    bb = ta_lib.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"]      = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["volume_ma_20"]  = df["volume"].rolling(20).mean()
    df["volume_ratio"]  = df["volume"] / df["volume_ma_20"]
    df["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(50).mean()) /
        df["volume"].rolling(50).std()
    )

    # ✅ Swing points — solo hacia atrás
    df["swing_high"]      = (df["high"] == df["high"].rolling(11).max()).astype(int)
    df["swing_low"]       = (df["low"]  == df["low"].rolling(11).min()).astype(int)
    df["last_swing_high"] = df["high"].where(df["swing_high"] == 1).ffill()
    df["last_swing_low"]  = df["low"].where(df["swing_low"]  == 1).ffill()

    # BOS
    df["bos_bullish"] = (
        (df["close"] > df["last_swing_high"].shift(1)) &
        (df["close"].shift(1) <= df["last_swing_high"].shift(1))
    ).astype(int)

    # ✅ Order Blocks — impulso ya ocurrió (sin shift(-1))
    body     = abs(df["close"] - df["open"])
    avg_body = body.rolling(20).mean()
    impulse  = body > avg_body * 1.5

    df["ob_bullish"] = (
        (df["close"].shift(1) < df["open"].shift(1)) &
        impulse &
        (df["close"] > df["open"])
    ).astype(int)
    df["ob_bull_top"]    = df["open"].shift(1).where(df["ob_bullish"] == 1).ffill()
    df["ob_bull_bottom"] = df["close"].shift(1).where(df["ob_bullish"] == 1).ffill()

    # ✅ FVG — confirmado con velas pasadas (sin shift(-1))
    df["fvg_bullish"]     = (df["high"].shift(2) < df["low"]).astype(int)
    df["fvg_bull_top"]    = df["low"].where(df["fvg_bullish"] == 1).ffill()
    df["fvg_bull_bottom"] = df["high"].shift(2).where(df["fvg_bullish"] == 1).ffill()

    # Kill zones
    hour = df["date"].dt.hour
    df["in_kill_zone"] = (
        ((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15))
    ).astype(int)

    # Features continuas SMC
    df["dist_to_ob"]         = ((df["close"] - df["ob_bull_top"]) / df["close"]).fillna(0)
    df["ob_size"]            = ((df["ob_bull_top"] - df["ob_bull_bottom"]) / df["close"]).fillna(0)
    df["dist_to_fvg"]        = ((df["close"] - df["fvg_bull_top"]) / df["close"]).fillna(0)
    df["fvg_size"]           = ((df["fvg_bull_top"] - df["fvg_bull_bottom"]) / df["close"]).fillna(0)
    df["dist_to_swing_high"] = ((df["last_swing_high"] - df["close"]) / df["close"]).fillna(0)
    df["dist_to_swing_low"]  = ((df["close"] - df["last_swing_low"]) / df["close"]).fillna(0)
    df["atr_norm"]           = df["atr"] / df["close"]

    ob_age, last_ob = [], np.nan
    for i in range(len(df)):
        if df["ob_bullish"].iloc[i] == 1:
            last_ob = i
        ob_age.append(i - last_ob if not np.isnan(last_ob) else 999)
    df["ob_age"] = ob_age

    # HTF merge
    df_1h = df_1h.copy()
    df_1h["ema_20_1h"]  = ta_lib.trend.ema_indicator(df_1h["close"], window=20)
    df_1h["ema_50_1h"]  = ta_lib.trend.ema_indicator(df_1h["close"], window=50)
    df_1h["ema_200_1h"] = ta_lib.trend.ema_indicator(df_1h["close"], window=200)
    df_1h["atr_1h"]     = ta_lib.volatility.average_true_range(
        df_1h["high"], df_1h["low"], df_1h["close"], window=14)
    df_1h["htf_bias"] = 0
    df_1h.loc[
        (df_1h["ema_20_1h"] > df_1h["ema_50_1h"]) &
        (df_1h["ema_50_1h"] > df_1h["ema_200_1h"]), "htf_bias"] = 1
    df_1h.loc[
        (df_1h["ema_20_1h"] < df_1h["ema_50_1h"]) &
        (df_1h["ema_50_1h"] < df_1h["ema_200_1h"]), "htf_bias"] = -1
    df_1h["htf_ema20_50_dist"]     = (df_1h["ema_20_1h"] - df_1h["ema_50_1h"]) / df_1h["ema_50_1h"]
    df_1h["htf_ema50_200_dist"]    = (df_1h["ema_50_1h"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]
    df_1h["htf_price_ema200_dist"] = (df_1h["close"] - df_1h["ema_200_1h"]) / df_1h["ema_200_1h"]

    df = pd.merge_asof(
        df.sort_values("date"),
        df_1h[["date","atr_1h","htf_bias","htf_ema20_50_dist",
               "htf_ema50_200_dist","htf_price_ema200_dist"]].sort_values("date"),
        on="date", direction="backward"
    )
    df["atr_ratio"] = df["atr"] / df["atr_1h"].replace(0, np.nan)

    return df.dropna(subset=["rsi", "atr", "htf_bias"])

def build_training_set(trades, features_df):
    trades = trades.copy()
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True)
    trades["target"]    = (trades["exit_reason"] == "roi").astype(int)
    features_df = features_df.set_index("date")

    rows, skipped = [], 0
    for _, trade in trades.iterrows():
        available = features_df.index[features_df.index <= trade["open_date"]]
        if len(available) == 0:
            skipped += 1
            continue
        candle = features_df.loc[available[-1]]
        row = {col: candle.get(col, 0) for col in FEATURE_COLS}
        row["target"]      = trade["target"]
        row["open_date"]   = trade["open_date"]
        row["exit_reason"] = trade["exit_reason"]
        rows.append(row)

    df_train = pd.DataFrame(rows).sort_values("open_date")
    print(f"\n📊 Dataset: {len(df_train)} muestras | ROI: {df_train['target'].sum()} | No-ROI: {(df_train['target']==0).sum()} | Balance: {df_train['target'].mean():.1%}")
    return df_train[FEATURE_COLS].fillna(0), df_train["target"], df_train["open_date"]

def train_model(X, y, dates):
    print("\n🤖 Walk-forward training...")
    tscv, scaler, fold_aucs = TimeSeriesSplit(n_splits=3), RobustScaler(), []

    for fold, (tr, te) in enumerate(tscv.split(X)):
        Xtr = scaler.fit_transform(X.iloc[tr])
        Xte = scaler.transform(X.iloc[te])
        ytr, yte = y.iloc[tr], y.iloc[te]
        pw = (ytr==0).sum() / max((ytr==1).sum(), 1)

        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=pw, eval_metric="logloss",
                               random_state=42, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

        yp   = m.predict_proba(Xte)[:, 1]
        auc  = roc_auc_score(yte, yp)
        fold_aucs.append(auc)
        print(f"\n  Fold {fold+1} — AUC: {auc:.3f}")
        print(classification_report(yte, (yp>=0.60).astype(int),
              target_names=["No-ROI","ROI"], zero_division=0))

    print(f"\n📈 AUC promedio: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

    # Modelo final
    Xf = scaler.fit_transform(X)
    pw = (y==0).sum() / max((y==1).sum(), 1)
    final = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=pw, eval_metric="logloss",
                               random_state=42, verbosity=0)
    final.fit(Xf, y)

    imp = pd.Series(final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n📊 Feature Importance (top 10):")
    print(imp.head(10).to_string())

    joblib.dump(final,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✅ Modelo: {MODEL_PATH}")
    print(f"✅ Scaler: {SCALER_PATH}")
    return final, scaler, imp

if __name__ == "__main__":
    print("=" * 60)
    print("  SMC ML Model Trainer — Anti-Leakage v2")
    print("=" * 60)
    trades        = load_backtest_trades()
    df_15m, df_1h = load_ohlcv()
    features_df   = compute_features(df_15m, df_1h)
    X, y, dates   = build_training_set(trades, features_df)

    if len(X) < 30:
        print(f"⚠️  Solo {len(X)} trades. Mínimo recomendado: 100.")
    else:
        train_model(X, y, dates)
        print("\n🎯 Listo. Ahora corre el backtest con el modelo actualizado.")