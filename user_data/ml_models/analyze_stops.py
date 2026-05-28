# analyze_stops.py — Autopsia completa: contexto de entrada + análisis trailing
import pandas as pd
import numpy as np
import json
import zipfile
import glob
from pathlib import Path
import ta

BACKTEST_PATH    = Path("user_data/backtest_results")
DATA_PATH        = Path("user_data/data/bybit")
PAIR             = "BTC_USDT"
TRAILING_OFFSET  = 0.015  # trailing_stop_positive_offset
TRAILING_POSITIVE = 0.01  # trailing_stop_positive

def load_trades():
    files = sorted(glob.glob(str(BACKTEST_PATH / "*.zip")))
    with zipfile.ZipFile(files[-1]) as z:
        json_files = [f for f in z.namelist() if f.endswith(".json")]
        with z.open(json_files[0]) as f:
            data = json.load(f)
    key = list(data["strategy"].keys())[0]
    trades = pd.DataFrame(data["strategy"][key]["trades"])
    trades["open_date"]  = pd.to_datetime(trades["open_date"], utc=True)
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True)
    return trades

def load_ohlcv():
    df = pd.read_feather(DATA_PATH / f"{PAIR}-15m.feather")
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df.sort_values("date").reset_index(drop=True)

def get_context_at_entry(df_ohlcv, entry_time):
    mask = df_ohlcv["date"] <= entry_time
    if mask.sum() < 20:
        return {}
    idx   = df_ohlcv[mask].index[-1]
    close = df_ohlcv["close"].iloc[:idx+1]
    high  = df_ohlcv["high"].iloc[:idx+1]
    low   = df_ohlcv["low"].iloc[:idx+1]

    rsi   = ta.momentum.rsi(close, window=14).iloc[-1]
    atr   = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]
    ema9  = close.ewm(span=9).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]
    ema50 = close.ewm(span=50).mean().iloc[-1]

    returns   = close.pct_change().dropna()
    vol_20    = returns.iloc[-20:].std() * np.sqrt(96)
    vol_100   = returns.iloc[-100:].std() * np.sqrt(96) if len(returns) >= 100 else vol_20
    vol_ratio = vol_20 / vol_100 if vol_100 > 0 else 1.0
    hour      = entry_time.hour

    return {
        "rsi":         round(rsi, 1),
        "atr_pct":     round(atr / close.iloc[-1] * 100, 3),
        "ema9_vs_21":  round((ema9  - ema21) / ema21 * 100, 3),
        "ema21_vs_50": round((ema21 - ema50) / ema50 * 100, 3),
        "vol_ratio":   round(vol_ratio, 2),
        "hour_utc":    hour,
        "in_london":   1 if 7  <= hour < 10 else 0,
        "in_ny":       1 if 12 <= hour < 15 else 0,
    }

if __name__ == "__main__":
    trades = load_trades()
    ohlcv  = load_ohlcv()

    stops   = trades[trades["exit_reason"] == "stop_loss"].copy()
    winners = trades[trades["exit_reason"].isin(["roi", "trailing_stop_loss"])].copy()

    print(f"Stop losses: {len(stops)} | Winners: {len(winners)}\n")

    # ── Sección 1: Contexto de entrada ───────────────────────────────────────
    stop_contexts = []
    for _, t in stops.iterrows():
        ctx = get_context_at_entry(ohlcv, t["open_date"])
        ctx["profit_pct"] = round(t["profit_ratio"] * 100, 3)
        ctx["duration_h"] = round((t["close_date"] - t["open_date"]).total_seconds() / 3600, 1)
        stop_contexts.append(ctx)

    win_contexts = []
    for _, t in winners.iterrows():
        ctx = get_context_at_entry(ohlcv, t["open_date"])
        ctx["profit_pct"] = round(t["profit_ratio"] * 100, 3)
        ctx["duration_h"] = round((t["close_date"] - t["open_date"]).total_seconds() / 3600, 1)
        win_contexts.append(ctx)

    df_stops = pd.DataFrame(stop_contexts)
    df_wins  = pd.DataFrame(win_contexts)

    print("═" * 58)
    print("  COMPARACIÓN: STOP LOSS vs WINNERS")
    print("═" * 58)

    metrics = ["rsi", "atr_pct", "ema9_vs_21", "ema21_vs_50", "vol_ratio", "duration_h"]
    print(f"\n{'Métrica':<20} {'Stop Loss':>12} {'Winners':>12} {'Diferencia':>12}")
    print("-" * 58)
    for m in metrics:
        if m in df_stops.columns and m in df_wins.columns:
            sv = df_stops[m].mean()
            wv = df_wins[m].mean()
            print(f"{m:<20} {sv:>12.3f} {wv:>12.3f} {sv-wv:>+12.3f}")

    print(f"\nDistribución por hora UTC (stops):")
    print(df_stops["hour_utc"].value_counts().sort_index().to_string())

    print(f"\nKill zone — stops:")
    print(f"  London (7-10h):  {df_stops['in_london'].sum()} / {len(df_stops)}")
    print(f"  NY     (12-15h): {df_stops['in_ny'].sum()} / {len(df_stops)}")

    print(f"\nKill zone — winners:")
    print(f"  London (7-10h):  {df_wins['in_london'].sum()} / {len(df_wins)}")
    print(f"  NY     (12-15h): {df_wins['in_ny'].sum()} / {len(df_wins)}")

    print(f"\n⚠️  Stops con RSI > 60:         {(df_stops['rsi'] > 60).sum()} / {len(df_stops)}")
    print(f"⚠️  Stops con vol_ratio > 1.5:  {(df_stops['vol_ratio'] > 1.5).sum()} / {len(df_stops)}")
    print(f"⚠️  Stops fuera de kill zone:   {(df_stops['in_london'] + df_stops['in_ny'] == 0).sum()} / {len(df_stops)}")
    print(f"\nDuración promedio stops:   {df_stops['duration_h'].mean():.1f}h")
    print(f"Duración promedio winners: {df_wins['duration_h'].mean():.1f}h")

    # ── Sección 2: Análisis trailing ─────────────────────────────────────────
    print("\n\n" + "═" * 80)
    print("  AUTOPSIA TRAILING: ¿el precio llegó al offset antes del stop?")
    print("═" * 80)
    print(f"\nConfig trailing: offset={TRAILING_OFFSET*100}% | positive={TRAILING_POSITIVE*100}%\n")
    print(f"{'#':<4} {'Entrada':>12} {'Max':>12} {'SL':>12} {'Max Exc%':>10} {'Llegó?':>8}  Categoría")
    print("-" * 85)

    never_reached   = []
    trailing_failed = []

    for i, (_, t) in enumerate(stops.iterrows(), 1):
        open_rate         = t["open_rate"]
        max_rate          = t["max_rate"]
        sl_abs            = t["stop_loss_abs"]
        max_excursion_pct = (max_rate - open_rate) / open_rate
        reached_offset    = max_excursion_pct >= TRAILING_OFFSET

        if reached_offset:
            trailing_failed.append(t)
            category = "⚠️  Trailing activado — no protegió"
        else:
            never_reached.append(t)
            category = "❌ Nunca llegó al +1.5%"

        print(f"{i:<4} {open_rate:>12.2f} {max_rate:>12.2f} {sl_abs:>12.2f} "
              f"{max_excursion_pct*100:>9.3f}% {str(reached_offset):>8}  {category}")

    print("\n" + "═" * 80)
    print(f"\n📊 RESUMEN TRAILING:")
    print(f"\n  ❌ Nunca llegaron al offset (+1.5%): {len(never_reached)} / {len(stops)}")
    print(f"     → El SL fijo de -2% los mató antes de tener profit real.")
    print(f"     → Solución potencial: SL más amplio o filtro de entrada más selectivo.")

    print(f"\n  ⚠️  Trailing activado pero no protegió: {len(trailing_failed)} / {len(stops)}")
    print(f"     → El precio llegó a +1.5% pero revirtió hasta tocar el SL.")
    print(f"     → Solución potencial: trailing más agresivo o salida por momentum.")

    if never_reached:
        df_nr = pd.DataFrame(never_reached)
        durations  = (df_nr["trade_duration"] / 60).tolist()
        excursions = [((t["max_rate"] - t["open_rate"]) / t["open_rate"] * 100) for _, t in df_nr.iterrows()]
        print(f"\n  Stats grupo 'nunca llegó':")
        print(f"    Duración promedio:     {sum(durations)/len(durations):.1f}h")
        print(f"    Max excursión promedio: {sum(excursions)/len(excursions):.3f}%")
        print(f"    Max excursión máxima:   {max(excursions):.3f}%")

    if trailing_failed:
        df_tf = pd.DataFrame(trailing_failed)
        durations  = (df_tf["trade_duration"] / 60).tolist()
        excursions = [((t["max_rate"] - t["open_rate"]) / t["open_rate"] * 100) for _, t in df_tf.iterrows()]
        print(f"\n  Stats grupo 'trailing fallido':")
        print(f"    Duración promedio:     {sum(durations)/len(durations):.1f}h")
        print(f"    Max excursión promedio: {sum(excursions)/len(excursions):.3f}%")
        print(f"    Max excursión máxima:   {max(excursions):.3f}%")