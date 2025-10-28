# BACKTESTING 
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import mlflow

default_outdir = Path("outputs")
default_outdir.mkdir(exist_ok=True)

@dataclass
class BacktestParams:
    sl: float = 0.014         
    tp: float = 0.014               
    shares: int = 0                
    commission_rt: float = 0.0125   
    borrow_rate_annual: float = 0.0025 
    initial_capital: float = 1000000.0 

def _apply_sl_tp_long(entry, hi, lo, close, tp, sl):
    """
    Devuelve el 'exit_price' aplicando primero TP y luego SL intradía para una posición LARGA.
    Si no toca niveles, se sale al close.
    """
    target = entry * (1 + tp)
    stop   = entry * (1 - sl)
    if hi >= target:           # toca TP
        return target
    if lo <= stop:             # toca SL
        return stop
    return close               # salida por close si no toca niveles

def _apply_sl_tp_short(entry, hi, lo, close, tp, sl):
    """
    Igual que arriba, pero para CORTOS (precio a favor = Bajar).
    """
    target = entry * (1 - tp)  # TP en cortos es precio más bajo
    stop   = entry * (1 + sl)  # SL en cortos es precio más alto
    if lo <= target:           # toca TP (bajó lo suficiente)
        return target
    if hi >= stop:             # toca SL (subió en contra)
        return stop
    return close               # salida por close

def backtest_signals_ohlc(df_prices: pd.DataFrame,
                          signals: pd.Series,
                          params: BacktestParams, outdir:Path |str = default_outdir, save_plots: bool=True) -> dict:
    """
    Estrategia diaria 'flat' al cierre:
    - Entra al CIERRE de t-1 si signal[t-1] != 0 y opera durante el día t con SL/TP por High/Low.
    - Sale el MISMO día t (flat overnight).
    - Comisión 0.125% round-trip por notional de entrada.
    - Borrow diario para cortos: 0.25%/252 * notional.
    Retorna métricas, equity y log de trades.
    """
    if isinstance(outdir, str):
        outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    # Orden y alineación
    d = df_prices.sort_values("date").reset_index(drop=True)
    sig = signals.reset_index(drop=True)
    assert len(sig) >= len(d) - 1 or len(sig) == len(d), \
        "Alinea signals con df_prices: debe haber al menos len(df)-1 señales."

    # Recorta/expande señales a longitud (len(d) - 1) pues cada trade usa (t-1)->t
    if len(sig) > len(d) - 1:
        sig = sig.iloc[:len(d)-1]
    elif len(sig) < len(d) - 1:
        # rellena con 0s al final si faltan
        sig = sig.reindex(range(len(d)-1), fill_value=0)

    # Inicialización
    equity = params.initial_capital
    equity_series = [equity]
    rets = []          # retornos diarios (sobre equity)
    trades = []        # log de operaciones

    for t in range(1, len(d)):
        s = int(sig.iloc[t-1])  # decisión tomada al cierre de t-1
        prev = d.loc[t-1]
        curr = d.loc[t]

        # si no hay trade, retorno 0
        if s == 0 or any(pd.isna(prev[c]) for c in ["Close"]) or any(pd.isna(curr[c]) for c in ["Close","High","Low"]):
            rets.append(0.0)
            equity_series.append(equity)
            continue

        entry = float(prev["Close"])
        hi, lo, cl = float(curr["High"]), float(curr["Low"]), float(curr["Close"])

        # Precio de salida con SL/TP intradía
        if s > 0:  # largo
            exit_price = _apply_sl_tp_long(entry, hi, lo, cl, params.tp, params.sl)
            pnl_gross = params.shares * (exit_price - entry)
            fees_commission = params.commission_rt * params.shares * entry  # round-trip %
            fees_borrow = 0.0  # sin préstamo en largos
        else:      # corto
            exit_price = _apply_sl_tp_short(entry, hi, lo, cl, params.tp, params.sl)
            pnl_gross = params.shares * (entry - exit_price)
            fees_commission = params.commission_rt * params.shares * entry
            fees_borrow = (params.borrow_rate_annual / 252.0) * params.shares * entry

        pnl_net = pnl_gross - fees_commission - fees_borrow

        # Actualizamos equity y retorno diario
        prev_eq = equity
        equity = equity + pnl_net
        daily_ret = (equity - prev_eq) / prev_eq if prev_eq > 0 else 0.0

        rets.append(daily_ret)
        equity_series.append(equity)

        trades.append({
            "date_entry": prev["date"],
            "date_exit": curr["date"],
            "side": "LONG" if s > 0 else "SHORT",
            "entry": entry,
            "exit": exit_price,
            "shares": params.shares,
            "pnl_gross": pnl_gross,
            "fees_commission": fees_commission,
            "fees_borrow": fees_borrow,
            "pnl_net": pnl_net
        })

    # DataFrames de resultados
    eq_df = pd.DataFrame({
        "date": d.loc[:, "date"].iloc[:len(equity_series)].values,
        "equity": equity_series
    })
    ret_df = pd.DataFrame({
        "date": d.loc[:, "date"].iloc[1:len(rets)+1].values,
        "ret": rets
    })
    trades_df = pd.DataFrame(trades)

    # Métricas
    N = len(rets)
    af = 252
    mean = np.mean(rets) if N else 0.0
    std = np.std(rets, ddof=1) if N > 1 else 0.0
    downside = np.array([r for r in rets if r < 0.0])
    dstd = np.std(downside, ddof=1) if len(downside) > 1 else 0.0

    sharpe = (mean / std) * np.sqrt(af) if std > 0 else np.nan
    sortino = (mean / dstd) * np.sqrt(af) if dstd > 0 else np.nan

    roll_max = np.maximum.accumulate(eq_df["equity"].values)
    dd = eq_df["equity"].values / roll_max - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    years = max(1e-9, (len(eq_df) - 1) / 252.0)
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0])**(1 / years) - 1 if len(eq_df) > 1 else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan
    win_rate = float((trades_df["pnl_net"] > 0).mean()) if len(trades_df) else np.nan

    # Guardados opcionales
    if save_plots:
        eq_path = outdir / "equity_curve.png"
        fig = plt.figure(figsize=(7,3))
        plt.plot(eq_df["date"], eq_df["equity"])
        plt.title("Equity Curve (reglas del proyecto)")
        plt.tight_layout()
        fig.savefig(eq_path, dpi=150)
        plt.close(fig)

        eq_csv = os.path.join(outdir, "equity_curve.csv")
        trades_csv = os.path.join(outdir, "trades.csv")
        ret_csv = os.path.join(outdir, "daily_returns.csv")
        eq_df.to_csv(eq_csv, index=False)
        trades_df.to_csv(trades_csv, index=False)
        ret_df.to_csv(ret_csv, index=False)

    # (Opcional) log a MLflow si lo estás usando en este script
    try:
        mlflow.log_artifact(eq_path)
        mlflow.log_artifact(eq_csv)
        mlflow.log_artifact(trades_csv)
        mlflow.log_artifact(ret_csv)
        mlflow.log_metrics({
            "bt_sharpe": float(sharpe),
            "bt_sortino": float(sortino),
            "bt_calmar": float(calmar),
            "bt_max_drawdown": float(max_dd),
            "bt_win_rate": float(win_rate),
            "bt_cagr": float(cagr),
        })
    except Exception:
        pass

    return {
        "metrics": {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "cagr": float(cagr),
            "final_equity": float(eq_df["equity"].iloc[-1]),
        },
        "equity": eq_df,
        "returns": ret_df,
        "trades": trades_df,
    }
