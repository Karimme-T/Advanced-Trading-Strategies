import os
from pathlib import Path
from dataclasses import dataclass
import sys
import subprocess


import numpy as np
import pandas as pd
import tensorflow as tf

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

from utils import backtest_model_on_splits, visualize_backtest_metrics, plot_portfolio_evolution, load_models

import mlflow
from mlflow.tracking import MlflowClient

model_name = "amzn_trading_model"

# CARGAR DESDE PRODUCTION STAGE
try:
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.keras.load_model(model_uri)
    
    # Obtener metadata del modelo
    client = MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    
    print(f"Modelo cargado desde Model Registry")
    print(f"   Nombre: {model_name}")
    print(f"   Versión: {model_version.version}")
    print(f"   Stage: Production")
    print(f"   Run ID: {model_version.run_id}")
    
except Exception as e:
    print(f" Error cargando modelo desde Registry: {e}")
    print("Intentando cargar desde archivo local...")
    
    # Fallback: cargar desde outputs/
    import tensorflow as tf
    model = tf.keras.models.load_model('outputs/best_mlp.keras')
    print("Modelo cargado desde archivo local")

def main():
    # Cargar mejores modelos entrenados
    # Cargar modelos (entrena si no existen)
    model_mlp, model_cnn = load_models()


    # Construir mapping de etiquetas 
    unique_labels = sorted(train_scaled["signal"].unique())        
    label_mapping = {lab: i for i, lab in enumerate(unique_labels)}  
    inv_mapping   = {v: k for k, v in label_mapping.items()}       

    feat_cols_model = list(feat_cols)


    mean_model = mean[feat_cols_model]
    std_model  = std[feat_cols_model].replace(0, 1.0)

    # Parámetros de backtesting 
    bt_params = BacktestParams(
        sl=0.014,
        tp=0.04,
        shares=100,           
        commission_rt=0.00125,   
        borrow_rate_annual=0.025,
        initial_capital=10000.0
    )

    # Backtest por splits para MLP y CNN
    results_mlp = backtest_model_on_splits(
        model=model_mlp,
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params=bt_params,
        outdir_root="outputs"
    )

    results_cnn = backtest_model_on_splits(
        model=model_cnn,
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params=bt_params,
        outdir_root="outputs"
    )

    # Imprimir resultados
    print("Backtest MLP:", {k: v["metrics"] for k, v in results_mlp.items()})
    print("Backtest CNN:", {k: v["metrics"] for k, v in results_cnn.items()})

    viz_path = Path("outputs/backtest_metrics_viz.html") 
    _ = visualize_backtest_metrics(results_mlp, results_cnn, viz_path)

    # Visualización de evolución del portafolio
    print("\nMostrando evolución del portafolio...")
    fig = plot_portfolio_evolution(results_mlp, results_cnn)
    fig.show()

    return {
        "mlp": results_mlp,
        "cnn": results_cnn
    }


if __name__ == "__main__":
    _ = main()
