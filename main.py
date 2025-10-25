"""
Pipeline principal - Ejecuta feature engineering, entrenamiento y backtesting
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from backtesting import backtest_signals_ohlc, BacktestParams

# Importar datos procesados y modelos entrenados
from Feature_eng import (
    data_with_y, 
    train_scaled, 
    val_scaled, 
    test_scaled,
    feat_cols,
    mean,
    std
)


def generate_signals_cfa(model, df_data, feat_cols, mean, std, inv_mapping, confidence_threshold=0.5):
    """
    Genera señales con política CFA lineal
    """
    df = df_data.copy()
    
    # Normalizar features
    features = df[feat_cols].copy()
    features = (features - mean) / std
    
    # Predicciones
    X = features.values.astype(np.float32)
    pred_probas = model.predict(X, verbose=0)
    
    signals = []
    for pred_proba in pred_probas:
        max_prob = pred_proba.max()
        pred_class = pred_proba.argmax()
        base_signal = inv_mapping[pred_class]
        
        # CFA forma lineal
        if max_prob >= confidence_threshold:
            confidence_factor = (max_prob - confidence_threshold) / (1 - confidence_threshold)
            weighted_signal = base_signal * confidence_factor
        else:
            weighted_signal = 0
        
        signals.append(int(np.sign(weighted_signal)))
    
    return pd.Series(signals, index=df.index)


def backtest_model_on_splits(model, inv_mapping, bt_params):
    """
    Ejecuta backtesting en train, val, test y full
    """
    results = {}
    
    # Train
    signals_train = generate_signals_cfa(model, train_scaled, feat_cols, mean, std, inv_mapping)
    df_train = train_scaled[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['train'] = backtest_signals_ohlc(df_train, signals_train, bt_params)
    
    # Val
    signals_val = generate_signals_cfa(model, val_scaled, feat_cols, mean, std, inv_mapping)
    df_val = val_scaled[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['val'] = backtest_signals_ohlc(df_val, signals_val, bt_params)
    
    # Test
    signals_test = generate_signals_cfa(model, test_scaled, feat_cols, mean, std, inv_mapping)
    df_test = test_scaled[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['test'] = backtest_signals_ohlc(df_test, signals_test, bt_params)
    
    # Full
    signals_full = generate_signals_cfa(model, data_with_y, feat_cols, mean, std, inv_mapping)
    df_full = data_with_y[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
    results['full'] = backtest_signals_ohlc(df_full, signals_full, bt_params)
    
    return results


def main():
    # 1️⃣ Cargar mejor modelo entrenado
    # Asumir que redes_version2.py ya corrió y guardó los modelos
    model_mlp = tf.keras.models.load_model("outputs/best_mlp.h5")
    model_cnn = tf.keras.models.load_model("outputs/best_cnn.h5")
    
    # Label mapping (de redes_version2.py)
    unique_labels = sorted(train_scaled["signal"].unique())
    label_mapping = {lab: i for i, lab in enumerate(unique_labels)}
    inv_mapping = {v: k for k, v in label_mapping.items()}
    
    # 2️⃣ Parámetros de backtesting
    bt_params = BacktestParams(
        sl=0.02,
        tp=0.04,
        shares=100,
        commission_rt=0.000125,
        borrow_rate_annual=0.025,
        initial_capital=10000.0
    )
    
    # 3️⃣ Backtest MLP
    results_mlp = backtest_model_on_splits(model_mlp, inv_mapping, bt_params)
    
    # 4️⃣ Backtest CNN
    results_cnn = backtest_model_on_splits(model_cnn, inv_mapping, bt_params)
    
    # 5️⃣ Retornar resultados
    return {
        'mlp': results_mlp,
        'cnn': results_cnn
    }


if __name__ == "__main__":
    results = main()