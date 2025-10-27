import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px

from backtesting import BacktestParams, backtest_signals_ohlc
from Feature_eng import (
    data_with_y,  
    train_scaled,  
    val_scaled,
    test_scaled,
    feat_cols,      
    mean,           
    std            
)


def generate_signals_cfa(
    model: tf.keras.Model,
    df_data: pd.DataFrame,
    feat_cols_model: list,
    mean_model: pd.Series,
    std_model: pd.Series,
    inv_mapping: dict,
    confidence_threshold: float = 0.5
) -> pd.Series:
    """
    Genera señales {-1,0,1} a partir de un modelo MLP (input 2D) o CNN 1D (input 3D).
    - Normaliza SOLO las columnas de 'feat_cols_model' con mean/std del TRAIN.
    - Para CNN: arma ventanas con lookback = model.input_shape[1] usando make_sequences
      sobre un DF ya normalizado.
    - Alinea la serie de señales con el índice original del df (rellena prefijo sin ventana con 0).
    """
    df = df_data.copy()

    # Normalización solo features del modelo
    feats = df.reindex(columns=feat_cols_model)
    feats_norm = (feats - mean_model) / std_model.replace(0, 1.0)

    # Detectar si el modelo espera 2D (MLP) o 3D (CNN 1D)
    in_shape = getattr(model, "input_shape", None)
    if in_shape is None:
        raise ValueError("No pude leer input_shape del modelo.")

    is_sequence_model = (len(in_shape) == 3)
    signals = np.zeros(len(df), dtype=int)

    if not is_sequence_model:

        X = feats_norm.values.astype(np.float32)
        pred_probas = model.predict(X, verbose=0)
        for i, pred_proba in enumerate(pred_probas):
            max_prob = float(np.max(pred_proba))
            pred_class = int(np.argmax(pred_proba)) 
            base_signal = int(inv_mapping[pred_class])  
            if max_prob >= confidence_threshold:
                conf = (max_prob - confidence_threshold) / (1.0 - confidence_threshold + 1e-12)
                weighted = base_signal * conf
            else:
                weighted = 0.0
            signals[i] = int(np.sign(weighted))
    else:

        lookback = in_shape[1] or 30 

        df_norm = df.copy()
        df_norm[feat_cols_model] = feats_norm

        from Feature_eng import make_sequences
        X_seq, _ = make_sequences(df_norm, feat_cols_model, lookback)
        X_seq = X_seq.astype(np.float32)

        pred_probas = model.predict(X_seq, verbose=0)

        start = lookback - 1
        for k, pred_proba in enumerate(pred_probas):
            i = start + k
            max_prob = float(np.max(pred_proba))
            pred_class = int(np.argmax(pred_proba))
            base_signal = int(inv_mapping[pred_class])
            if max_prob >= confidence_threshold:
                conf = (max_prob - confidence_threshold) / (1.0 - confidence_threshold + 1e-12)
                weighted = base_signal * conf
            else:
                weighted = 0.0
            signals[i] = int(np.sign(weighted))

    return pd.Series(signals, index=df.index, name="signal_pred")


def backtest_model_on_splits(
    model: tf.keras.Model,
    inv_mapping: dict,
    feat_cols_model: list,
    mean_model: pd.Series,
    std_model: pd.Series,
    bt_params: BacktestParams,
    outdir_root: str | Path = "outputs"
) -> dict:
    """
    Ejecuta backtesting en train, val, test y full, usando:
    - Señales generadas sobre features normalizados (mean/std del TRAIN).
    - Precios crudos (OHLCV) para el motor de backtest.
    """
    outdir_root = Path(outdir_root)

    tr_idx = train_scaled.index
    va_idx = val_scaled.index
    te_idx = test_scaled.index

    train_raw = data_with_y.loc[tr_idx].copy()
    val_raw   = data_with_y.loc[va_idx].copy()
    test_raw  = data_with_y.loc[te_idx].copy()
    full_raw  = data_with_y.copy()

    results = {}

    # Train 
    signals_train = generate_signals_cfa(model, train_raw, feat_cols_model, mean_model, std_model, inv_mapping)
    df_train = train_raw[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['train'] = backtest_signals_ohlc(df_train, signals_train, bt_params, outdir=outdir_root / "bt_train")

    #  Val
    signals_val = generate_signals_cfa(model, val_raw, feat_cols_model, mean_model, std_model, inv_mapping)
    df_val = val_raw[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['val'] = backtest_signals_ohlc(df_val, signals_val, bt_params, outdir=outdir_root / "bt_val")

    # Test
    signals_test = generate_signals_cfa(model, test_raw, feat_cols_model, mean_model, std_model, inv_mapping)
    df_test = test_raw[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['test'] = backtest_signals_ohlc(df_test, signals_test, bt_params, outdir=outdir_root / "bt_test")

    # Full 
    signals_full = generate_signals_cfa(model, full_raw, feat_cols_model, mean_model, std_model, inv_mapping)
    df_full = full_raw[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['full'] = backtest_signals_ohlc(df_full, signals_full, bt_params, outdir=outdir_root / "bt_full")

    return results

def visualize_backtest_metrics(results_mlp: dict, results_cnn: dict, output_path: Path):
    """
    Crea una visualización Plotly de las métricas clave de backtesting.
    Compara MLP y CNN a través de los splits (train, val, test, full).
    Guarda el resultado como un archivo JSON de Plotly.
    """
    
    data = []
    # Procesar resultados MLP
    for split, res in results_mlp.items():
        metrics = res["metrics"]
        data.append({
            "Model": "MLP", "Split": split.capitalize(), 
            "Sharpe": metrics["sharpe"], "CAGR": metrics["cagr"], 
            "Max_Drawdown": metrics["max_drawdown"], "Win_Rate": metrics["win_rate"]
        })
    
    # Procesar resultados CNN
    for split, res in results_cnn.items():
        metrics = res["metrics"]
        data.append({
            "Model": "CNN", "Split": split.capitalize(), 
            "Sharpe": metrics["sharpe"], "CAGR": metrics["cagr"], 
            "Max_Drawdown": metrics["max_drawdown"], "Win_Rate": metrics["win_rate"]
        })

    df = pd.DataFrame(data)
    
    # Pivotear el dataframe para Plotly Express
    df_melt = df.melt(
        id_vars=["Model", "Split"], 
        value_vars=["Sharpe", "CAGR", "Max_Drawdown", "Win_Rate"],
        var_name="Metric",
        value_name="Value"
    )
    
    # Crear el gráfico de barras agrupado
    fig = px.bar(
        df_melt, 
        x="Split", 
        y="Value", 
        color="Model", 
        facet_col="Metric", 
        facet_col_wrap=2,
        title="Comparación de Métricas de Backtesting (MLP vs CNN)",
        barmode="group",
        category_orders={"Metric": ["Sharpe", "CAGR", "Max_Drawdown", "Win_Rate"]}
    )

    # Ajustar títulos para Max_Drawdown y Win_Rate
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Metric=", "")))
    
    print("Mostrando el gráfico interactivo. Puede abrirse en una nueva ventana/pestaña del navegador.")
    fig.show()

    return "Plotly figure shown"