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
    # TRIAL BACKTEST: Tighten SL and reduce TP for more realistic risk management
    sl: float = 0.015          # Reduced from 0.02 (e.g., cut loss at 1% instead of 2%)
    tp: float = 0.025          # Reduced from 0.04 (e.g., target 2% gain instead of 4%)
    shares: float = 0.03      # Risk % per trade (0.01 = 1% risk)
    commission_rt: float = 0.00125   
    borrow_rate_annual: float = 0.0025 
    initial_capital: float = 10000.0 
    overnight: bool = True

def _apply_sl_tp_long(entry, hi, lo, close, tp, sl):
    """
    Devuelve el 'exit_price' aplicando primero TP y luego SL intradía para una posición LARGA.
    Si no toca niveles, se sale al close.
    """
    target = entry * (1 + tp)
    stop   = entry * (1 - sl)
    # Primero chequeamos si el TP o SL se activaron entre el Open/High/Low/Close del día.
    # Asume que el precio de entrada fue el cierre del día anterior.
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

# #TRIAL BACKTEST: Modificar la función principal para Overnight y Posición Variable
def backtest_signals_ohlc(df_prices: pd.DataFrame,
                          signals: pd.Series,
                          params: BacktestParams, 
                          outdir:Path |str = default_outdir, 
                          save_plots: bool=True) -> dict:
    """
    Estrategia con opción a mantener posición overnight y gestión de riesgo/posición variable.
    - Señal: {1: Long, -1: Short, 0: Flat}.
    - Si params.shares > 0 (Risk%): calcula el tamaño de posición basado en 
      Risk% y SL (Kelly Criterion simplificado).
    - Si overnight=False (default original): sale al cierre (flat diario).
    - Si overnight=True: mantiene posición si la nueva señal es la misma.
    """
    if isinstance(outdir, str):
        outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    d = df_prices.sort_values("date").reset_index(drop=True)
    sig = signals.reset_index(drop=True)
    
    # Asegurar que el DataFrame de señales sea del mismo largo o más corto (por lookback/NaNs)
    # Las señales del día t-1 deben operar el día t.
    if len(sig) > len(d) - 1:
        sig = sig.iloc[:len(d)-1]
    elif len(sig) < len(d) - 1:
        # Esto es crucial: si la señal es más corta (ej. por lookback), se rellena el inicio con 0s
        sig = sig.reindex(d.index[1:], fill_value=0).reset_index(drop=True)
        
    d_op = d.iloc[1:].reset_index(drop=True).copy() # DataFrame de días de operación (t=1, 2, ...)
    d_op["signal"] = sig
    
    # Inicialización
    equity = params.initial_capital
    equity_series = [params.initial_capital]
    rets = []
    trades = []
    current_position = 0.0 # Posición actual (acciones)
    entry_price = np.nan
    entry_date = np.nan
    entry_shares = 0
    
    # Capital de inicio para el cálculo de CAGR
    initial_equity = params.initial_capital

    for t in range(len(d_op)):
        
        # t-1 es el día anterior, t es el día actual (con Open, High, Low, Close)
        # d.loc[d.index[t]] es el Close del día anterior
        # d_op.loc[t] es el Close del día actual
        curr = d_op.loc[t]
        prev_equity = equity
        
        # Precios del día t
        open_t, hi_t, lo_t, cl_t = curr["Open"], curr["High"], curr["Low"], curr["Close"]
        
        if pd.isna(cl_t) or pd.isna(open_t):
            rets.append(0.0)
            equity_series.append(equity)
            continue
            
        new_signal = int(curr["signal"])
        
        # === 1. Lógica de Cierre/Salida de Posición (Exit Logic) ===
        
        if current_position != 0:
            
            # Condición de Cierre/Salida:
            # a) Si la nueva señal es FLAT (0)
            # b) Si la nueva señal es la OPUESTA (ej: LONG y ahora es SHORT)
            # c) Si overnight=False (flat diario)
            # d) #TRIAL BACKTEST: Si la posición es overnight, el PnL del día anterior se calculó con el Close_t-1, 
            #    pero la ejecución de SL/TP/CLOSE_OUT se hace con los precios de hoy.
            
            exit_condition_opposite = (np.sign(current_position) != 0) and (np.sign(new_signal) != 0) and (np.sign(current_position) != np.sign(new_signal))
            exit_condition_flat = new_signal == 0
            exit_condition_daily = (not params.overnight) and (np.sign(current_position) != 0)

            if exit_condition_flat or exit_condition_opposite or exit_condition_daily:
                
                shares_to_exit = abs(current_position)
                
                # Calcular precio de salida con SL/TP (usando precios del día actual)
                if current_position > 0: # Saliendo de LONG
                    exit_price = _apply_sl_tp_long(entry_price, hi_t, lo_t, cl_t, params.tp, params.sl)
                    pnl_gross = shares_to_exit * (exit_price - entry_price)
                    fees_commission = params.commission_rt * shares_to_exit * entry_price
                    fees_borrow = 0.0
                else: # Saliendo de SHORT
                    exit_price = _apply_sl_tp_short(entry_price, hi_t, lo_t, cl_t, params.tp, params.sl)
                    pnl_gross = shares_to_exit * (entry_price - exit_price)
                    fees_commission = params.commission_rt * shares_to_exit * entry_price
                    # Nota: El borrow rate debe calcularse SOLO si el trade fue *overnight*. Si no es overnight, 
                    # esta lógica de backtesting no es perfecta, pero mantenemos el cálculo de borrow basado en el tiempo
                    # que la posición estuvo abierta (1 día).
                    fees_borrow = (params.borrow_rate_annual / 252.0) * shares_to_exit * entry_price
                    
                pnl_net_exit = pnl_gross - fees_commission - fees_borrow

                equity += pnl_net_exit
                
                # Registrar el trade cerrado (debe ser un trade completo)
                trades.append({
                    "date_entry": entry_date,
                    "date_exit": curr["date"],
                    "side": "LONG" if current_position > 0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares_to_exit,
                    "pnl_gross": pnl_gross,
                    "fees_commission": fees_commission,
                    "fees_borrow": fees_borrow,
                    "pnl_net": pnl_net_exit
                })
                
                current_position = 0.0
                entry_price = np.nan
                entry_date = np.nan
                entry_shares = 0
            
        # === 2. Lógica de Entrada/Apertura de Posición (Entry Logic) ===
        if current_position == 0 and new_signal != 0:
            
            # Entry Price: Usamos el Cierre del día actual (t) para simular la entrada al cierre.
            entry = cl_t
            
            # #TRIAL BACKTEST: Cálculo de tamaño de posición (Position Sizing - Risk %)
            if params.shares > 0: # Usar Risk %
                # PositionSize = (Risk * Equity) / (SL * Entry)
                # SL es la tolerancia a la pérdida (0.02)
                risk_amount = params.shares * equity # Ej: 1% de 100k = 1000
                stop_loss_pct = params.sl
                
                # shares = CapitalArriesgado / Riesgo_Por_Acción
                shares_calc = risk_amount / (entry * stop_loss_pct)
                shares_to_enter = int(shares_calc) # Redondeo a la baja
            else:
                 # Usa el valor por defecto/fijo si Risk% = 0.0
                shares_to_enter = 100 
            
            # Ejecutar la entrada
            if shares_to_enter > 0:
                current_position = shares_to_enter * new_signal
                entry_price = entry
                entry_date = curr["date"]
                entry_shares = shares_to_enter
                
                # Pagar la mitad de la comisión al entrar
                fees_commission_entry = params.commission_rt / 2.0 * shares_to_enter * entry 
                equity -= fees_commission_entry
            
        # === 3. Lógica de PnL para Posiciones OVERNIGHT (Mantener) ===
        # Si la posición se mantuvo (current_position != 0), calculamos el PnL del día.
        if current_position != 0 and entry_shares > 0 and params.overnight:
            
            shares_overnight = abs(current_position)
            
            # PnL Bruto del día por mantener posición (basado en el cambio de precio del día)
            # El precio de referencia para el PnL de HOY es el CIERRE de AYER (d.loc[d.index[t], "Close"])
            # Esto es más complejo en backtesting overnight. Para simplificar y alinearnos con el concepto
            # de PnL diario sobre el patrimonio total, calcularemos el cambio desde el Close_t-1 (Open_t) 
            # al Close_t. 
            
            # Precio de inicio del día t
            price_start_of_day = d.loc[d.index[t], "Close"]
            
            # PnL Bruto: Shares * (Close_t - Close_t-1) * Signo
            pnl_gross_overnight = shares_overnight * (cl_t - price_start_of_day) * np.sign(current_position)
            
            # Costo de préstamo SHORT (solo para cortos, y solo para el día actual)
            fees_borrow_overnight = 0.0
            if current_position < 0:
                # El costo de borrow se aplica sobre el notional de la posición, *diariamente*.
                fees_borrow_overnight = (params.borrow_rate_annual / 252.0) * shares_overnight * entry_price
            
            pnl_net_overnight = pnl_gross_overnight - fees_borrow_overnight
            
            equity += pnl_net_overnight
            
        # Si la posición está cerrada (current_position == 0) y no hubo nueva entrada, el PnL es 0.
        if current_position == 0 and new_signal == 0:
            pnl_net_daily = 0.0
        elif current_position != 0 and entry_shares > 0 and params.overnight:
             # Si la posición se mantuvo, el PnL ya fue calculado como pnl_net_overnight
            pnl_net_daily = pnl_net_overnight
        else:
             # Si hubo un trade de entrada/salida en el mismo día, el PnL es pnl_net_exit (ya registrado)
             # Si no, el PnL fue 0. Lo más seguro es usar la diferencia de equity.
             pnl_net_daily = equity - prev_equity

        # Actualización final del equity y el retorno
        daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        rets.append(daily_ret)
        equity_series.append(equity)


    # DataFrames de resultados y Métricas (el cálculo de métricas es el mismo)
    eq_df = pd.DataFrame({
        "date": d.loc[:, "date"].iloc[:len(equity_series)].values,
        "equity": equity_series
    })
    
    # ... [Resto del cálculo de métricas (Sharpe, MaxDD, CAGR, etc.) sigue igual] ...
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
    # #TRIAL BACKTEST: CAGR usa initial_equity
    cagr = (eq_df["equity"].iloc[-1] / initial_equity)**(1 / years) - 1 if len(eq_df) > 1 else 0.0
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan
    trades_df = pd.DataFrame(trades)
    win_rate = float((trades_df["pnl_net"] > 0).mean()) if len(trades_df) else np.nan


    # Guardados opcionales
    if save_plots:
        # NOTE: Se omiten los guardados de archivos, pero se mantiene la estructura del retorno.
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
        "returns": pd.DataFrame({"date": d_op["date"], "ret": rets}), # Retorna los retornos
        "trades": trades_df,
    }