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

from utils import backtest_model_on_splits, visualize_backtest_metrics

def main():
    # Cargar mejores modelos entrenados
    #model_mlp = tf.keras.models.load_model("outputs/best_mlp.keras", compile=False)
    #model_cnn = tf.keras.models.load_model("outputs/best_cnn.keras", compile=False)

    #TRIAL AUTOMAT REDES.PY EXECUTION

    model_mlp_path = "outputs/best_mlp.keras"
    model_cnn_path = "outputs/best_cnn.keras"

    if not os.path.exists(model_mlp_path) or not os.path.exists(model_cnn_path):
        print("⚠️  Modelos no encontrados. Ejecutando entrenamiento...")
        print(f"   MLP existe: {os.path.exists(model_mlp_path)}")
        print(f"   CNN existe: {os.path.exists(model_cnn_path)}")
        
        try:
            # Ejecutar redes.py
            result = subprocess.run(
                [sys.executable, "redes.py"],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Entrenamiento completado exitosamente")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error durante el entrenamiento:")
            print(e.stderr)
            raise RuntimeError("No se pudieron entrenar los modelos") from e
        
        # Verificar nuevamente que los modelos existan
        if not os.path.exists(model_mlp_path) or not os.path.exists(model_cnn_path):
            raise FileNotFoundError(
                f"Los modelos no se generaron correctamente después del entrenamiento.\n"
                f"MLP: {os.path.exists(model_mlp_path)}, CNN: {os.path.exists(model_cnn_path)}"
            )
    
    # Cargar mejores modelos entrenados
    print("📂 Cargando modelos...")
    try:
        model_mlp = tf.keras.models.load_model(model_mlp_path, compile=False)
        model_cnn = tf.keras.models.load_model(model_cnn_path, compile=False)
        print("✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"❌ Error al cargar los modelos: {e}")
        raise


    # END TRIAL AUTOMAT REDES.PY EXECUTION

    # Construir mapping de etiquetas 
    unique_labels = sorted(train_scaled["signal"].unique())        
    label_mapping = {lab: i for i, lab in enumerate(unique_labels)}  
    inv_mapping   = {v: k for k, v in label_mapping.items()}       

    feat_cols_model = list(feat_cols)


    mean_model = mean[feat_cols_model]
    std_model  = std[feat_cols_model].replace(0, 1.0)

    # Parámetros de backtesting 
    bt_params = BacktestParams(
        sl=0.02,
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

    return {
        "mlp": results_mlp,
        "cnn": results_cnn
    }


if __name__ == "__main__":
    _ = main()
