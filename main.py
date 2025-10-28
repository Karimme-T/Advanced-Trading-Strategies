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

from utils import backtest_model_on_splits, visualize_backtest_metrics, plot_portfolio_evolution, load_models, optimize_backtest_params

def main():
    # Cargar mejores modelos entrenados
    # Cargar modelos (entrena si no existen)
    model_mlp, model_cnn = load_models()
    # END TRIAL AUTOMAT REDES.PY EXECUTION

    # Construir mapping de etiquetas 
    unique_labels = sorted(train_scaled["signal"].unique())        
    label_mapping = {lab: i for i, lab in enumerate(unique_labels)}  
    inv_mapping   = {v: k for k, v in label_mapping.items()}       

    feat_cols_model = list(feat_cols)


    mean_model = mean[feat_cols_model]
    std_model  = std[feat_cols_model].replace(0, 1.0)

    # #TRIAL BACKTEST: Definir parámetros base (con gestión de riesgo)
    bt_params_base = BacktestParams(
        sl=0.02,
        tp=0.04,
        shares=0.01,           # 1% de riesgo por trade (Risk%)
        commission_rt=0.00125,   
        borrow_rate_annual=0.025,
        initial_capital=10000.0,
        overnight=True         # Permitir overnight
    )

    # #TRIAL BACKTEST: Optimización de Umbral y Shares usando VALIDATION set
    print("\n🚀 Optimizando Umbral de Confianza y Shares para CALMAR RATIO con Validation Set...") # TRIAL BACKTEST: Cambiado el mensaje
    
    # MLP Optimization
    optimized_mlp_params = optimize_backtest_params(
        model=model_mlp,
        df_data=val_scaled,  # Usar Validation Set
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params_base=bt_params_base,
        split_name="Val"
    )
    # CNN Optimization
    optimized_cnn_params = optimize_backtest_params(
        model=model_cnn,
        df_data=val_scaled,  # Usar Validation Set
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params_base=bt_params_base,
        split_name="Val",
        is_cnn=True
    )
    
    print("\n✅ Parámetros Optimizados:")
    # TRIAL BACKTEST: Incluir Calmar en el reporte
    print(f"MLP (Calmar={optimized_mlp_params.get('calmar', np.nan):.4f}): Conf={optimized_mlp_params['confidence_threshold']:.2f}, Shares/Risk={optimized_mlp_params['shares']:.4f}")
    print(f"CNN (Calmar={optimized_cnn_params.get('calmar', np.nan):.4f}): Conf={optimized_cnn_params['confidence_threshold']:.2f}, Shares/Risk={optimized_cnn_params['shares']:.4f}")

    # #TRIAL BACKTEST: Usar los parámetros optimizados para el backtest final
    
    # FIX for TypeError: Remove non-dataclass arguments (confidence_threshold, sharpe, calmar)
    params_to_remove = ['confidence_threshold', 'sharpe', 'calmar'] # TRIAL BACKTEST: Añadir 'calmar'
    params_mlp_to_pass = {k: v for k, v in optimized_mlp_params.items() if k not in params_to_remove}
    params_cnn_to_pass = {k: v for k, v in optimized_cnn_params.items() if k not in params_to_remove}

    # Re-instantiate BacktestParams using the filtered dictionaries
    bt_params_mlp = BacktestParams(
        **params_mlp_to_pass
    )
    bt_params_cnn = BacktestParams(
        **params_cnn_to_pass
    )
    # --- END OF FIX ---

    # Backtest por splits para MLP y CNN
    results_mlp = backtest_model_on_splits(
        model=model_mlp,
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params=bt_params_mlp, # Usar params optimizados
        outdir_root="outputs"
    )

    results_cnn = backtest_model_on_splits(
        model=model_cnn,
        inv_mapping=inv_mapping,
        feat_cols_model=feat_cols_model,
        mean_model=mean_model,
        std_model=std_model,
        bt_params=bt_params_cnn, # Usar params optimizados
        outdir_root="outputs"
    )

    # Imprimir resultados
    print("Backtest MLP:", {k: v["metrics"] for k, v in results_mlp.items()})
    print("Backtest CNN:", {k: v["metrics"] for k, v in results_cnn.items()})

    viz_path = Path("outputs/backtest_metrics_viz.html") 
    _ = visualize_backtest_metrics(results_mlp, results_cnn, viz_path)

    # Visualización de evolución del portafolio
    print("\n📊 Mostrando evolución del portafolio...")
    fig = plot_portfolio_evolution(results_mlp, results_cnn)
    fig.show()

    return {
        "mlp": results_mlp,
        "cnn": results_cnn
    }


if __name__ == "__main__":
    _ = main()
