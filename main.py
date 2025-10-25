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
    Genera señales {-1,0,1} a partir de un modelo MLP (input 2D) o CNN 1D (input 3D).
    - Normaliza con mean/std provistos.
    - Para CNN: arma ventanas con lookback = model.input_shape[1] usando make_sequences
      sobre un DF ya normalizado.
    - Alinea la serie de señales con el índice original del df (rellena el prefijo sin ventana con 0).
    """
    import numpy as np
    import pandas as pd

    df = df_data.copy()

    # Normalización consistente
    feats_norm = (df[feat_cols] - mean) / std

    # Detectar si el modelo espera 2D (MLP) o 3D (CNN)
    in_shape = getattr(model, "input_shape", None)
    if in_shape is None:
        raise ValueError("No pude leer input_shape del modelo.")

    is_sequence_model = (len(in_shape) == 3)  

    signals = np.zeros(len(df), dtype=int) 

    if not is_sequence_model:
        # MLP: input 2D 
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
        # CNN: input 3D 
        lookback = in_shape[1]
        if lookback is None:
            lookback = 30

        df_norm = df.copy()
        df_norm[feat_cols] = feats_norm


        from Feature_eng import make_sequences
        X_seq, _ = make_sequences(df_norm, feat_cols, lookback) 
        X_seq = X_seq.astype(np.float32)

        pred_probas = model.predict(X_seq, verbose=0)

        # Alinear: la predicción k corresponde a la fila índice (lookback-1 + k)
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

        # las primeras (lookback-1) posiciones quedan 0 (HOLD) para mantener longitud

    return pd.Series(signals, index=df.index, name="signal_pred")



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
    model_mlp = tf.keras.models.load_model("outputs/best_mlp.keras", compile=False)
    model_cnn = tf.keras.models.load_model("outputs/best_cnn.keras", compile=False)
    
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