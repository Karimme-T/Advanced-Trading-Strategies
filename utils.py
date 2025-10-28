import os
from pathlib import Path
from dataclasses import dataclass
import sys
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    Muestra 4 subplots (uno por split) con todas las métricas como barras agrupadas.
    """
    
    # Definir el orden de los splits
    split_order = ['train', 'val', 'test', 'full']
    metric_names = ['Sharpe', 'Sortino', 'Calmar', 'CAGR', 'Max_Drawdown', 'Win_Rate']
    
    # Procesar datos
    data = []
    
    # Procesar resultados MLP
    for split, res in results_mlp.items():
        metrics = res["metrics"]
        data.append({
            "Model": "MLP", 
            "Split": split.lower(), 
            "Sharpe": metrics["sharpe"], 
            "Sortino": metrics["sortino"],
            "Calmar": metrics["calmar"],
            "CAGR": metrics["cagr"], 
            "Max_Drawdown": metrics["max_drawdown"], 
            "Win_Rate": metrics["win_rate"]
        })
    
    # Procesar resultados CNN
    for split, res in results_cnn.items():
        metrics = res["metrics"]
        data.append({
            "Model": "CNN", 
            "Split": split.lower(), 
            "Sharpe": metrics["sharpe"], 
            "Sortino": metrics["sortino"],
            "Calmar": metrics["calmar"],
            "CAGR": metrics["cagr"], 
            "Max_Drawdown": metrics["max_drawdown"], 
            "Win_Rate": metrics["win_rate"]
        })

    df = pd.DataFrame(data)
    
    # Crear subplots (2x2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[split.capitalize() for split in split_order],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Colores para los modelos
    colors = {'MLP': '#1f77b4', 'CNN': '#ff7f0e'}
    
    # Posiciones de los subplots
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    # Crear barras para cada split
    for idx, split in enumerate(split_order):
        row, col = positions[idx]
        
        # Filtrar datos para este split
        df_split = df[df['Split'] == split]
        
        if df_split.empty:
            continue
        
        # Obtener datos para MLP y CNN
        mlp_data = df_split[df_split['Model'] == 'MLP']
        cnn_data = df_split[df_split['Model'] == 'CNN']
        
        # Preparar valores
        mlp_values = [mlp_data[metric].values[0] if len(mlp_data) > 0 else 0 for metric in metric_names]
        cnn_values = [cnn_data[metric].values[0] if len(cnn_data) > 0 else 0 for metric in metric_names]
        
        # Agregar barras MLP
        fig.add_trace(
            go.Bar(
                name='MLP',
                x=metric_names,
                y=mlp_values,
                marker_color=colors['MLP'],
                showlegend=(idx == 0),  # Solo mostrar leyenda en el primer subplot
                text=[f'{v:.3f}' for v in mlp_values],
                textposition='outside',
                textfont=dict(size=9)
            ),
            row=row, col=col
        )
        
        # Agregar barras CNN
        fig.add_trace(
            go.Bar(
                name='CNN',
                x=metric_names,
                y=cnn_values,
                marker_color=colors['CNN'],
                showlegend=(idx == 0),
                text=[f'{v:.3f}' for v in cnn_values],
                textposition='outside',
                textfont=dict(size=9)
            ),
            row=row, col=col
        )
        
        # Configurar ejes para este subplot
        fig.update_xaxes(tickangle=-45, row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    # Configuración general del layout
    fig.update_layout(
        title={
            'text': 'Comparación de Métricas de Backtesting por Split (MLP vs CNN)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        barmode='group',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white'
    )
    
    print("Mostrando el gráfico interactivo. Puede abrirse en una nueva ventana/pestaña del navegador.")
    fig.show()

    return "Plotly figure shown"

def plot_portfolio_evolution(results_mlp: dict, results_cnn: dict) -> go.Figure:
    """
    Crea un gráfico interactivo con Plotly mostrando la evolución del equity
    de ambos modelos (MLP y CNN) SOLO en los splits 'val' y 'test'.
    
    Args:
        results_mlp: Diccionario con resultados del backtest MLP {split: {"equity": df, ...}}
        results_cnn: Diccionario con resultados del backtest CNN {split: {"equity": df, ...}}
    
    Returns:
        go.Figure: Figura de Plotly con las curvas de equity
    """
    
    fig = go.Figure()
    
    # Splits a incluir
    target_splits = ['val', 'test']
    
    # Colores para los splits (Solo se usarán los dos primeros)
    colors_mlp = ['#1f77b4', '#aec7e8', '#c6dbef']  # Azules
    colors_cnn = ['#ff7f0e', '#ffbb78', '#fdd0a2']  # Naranjas
    
    # Diccionario para asignar colores consistentemente
    color_map_mlp = {'val': colors_mlp[0], 'test': colors_mlp[1]}
    color_map_cnn = {'val': colors_cnn[0], 'test': colors_cnn[1]}
    
    # Contador de iteración para el color, inicializado para 'val' y 'test'
    i = 0
    
    # Agregar curvas MLP - Filtrando solo 'val' y 'test'
    for split_name, result in results_mlp.items():
        if split_name in target_splits:
            equity_df = result["equity"]
            
            # Usar el mapa de colores definido para el split
            color = color_map_mlp[split_name]
            
            fig.add_trace(go.Scatter(
                x=equity_df["date"],
                y=equity_df["equity"],
                mode='lines',
                name=f'MLP - {split_name}',
                line=dict(color=color, width=2),
                hovertemplate='<b>MLP - %{fullData.name}</b><br>' +
                              'Fecha: %{x}<br>' +
                              'Equity: $%{y:,.2f}<extra></extra>'
            ))
            i += 1
    
    # Reiniciar contador/índice para los colores CNN si fuera necesario
    # i = 0 
    
    # Agregar curvas CNN - Filtrando solo 'val' y 'test'
    for split_name, result in results_cnn.items():
        if split_name in target_splits:
            equity_df = result["equity"]
            
            # Usar el mapa de colores definido para el split
            color = color_map_cnn[split_name]
            
            fig.add_trace(go.Scatter(
                x=equity_df["date"],
                y=equity_df["equity"],
                mode='lines',
                name=f'CNN - {split_name}',
                line=dict(color=color, width=2),
                hovertemplate='<b>CNN - %{fullData.name}</b><br>' +
                              'Fecha: %{x}<br>' +
                              'Equity: $%{y:,.2f}<extra></extra>'
            ))
            i += 1
    
    # Línea de capital inicial (referencia) - Se mantiene la lógica de rango de fechas
    if results_mlp or results_cnn:
        # Usar cualquier resultado para obtener el capital inicial
        first_result = next((r for r in list(results_mlp.values()) + list(results_cnn.values()) if r.get("equity") is not None), None)

        if first_result:
            initial_capital = first_result["equity"]["equity"].iloc[0]
            
            # Obtener rango de fechas completo
            all_dates = []
            for split_name in target_splits:
                if split_name in results_mlp:
                    all_dates.extend(results_mlp[split_name]["equity"]["date"].tolist())
                if split_name in results_cnn:
                    all_dates.extend(results_cnn[split_name]["equity"]["date"].tolist())
            
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                
                fig.add_trace(go.Scatter(
                    x=[min_date, max_date],
                    y=[initial_capital, initial_capital],
                    mode='lines',
                    name='Capital Inicial',
                    line=dict(color='gray', width=1, dash='dot'),
                    hovertemplate='Capital Inicial: $%{y:,.2f}<extra></extra>'
                ))
    
    # Configuración del layout (se mantiene igual)
    fig.update_layout(
        title={
            'text': 'Evolución del Portafolio - MLP vs CNN (Validation y Test)', # Título actualizado
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Fecha',
        yaxis_title='Equity ($)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#cccccc',
            borderwidth=1
        ),
        height=600,
        margin=dict(l=80, r=40, t=80, b=60),
        yaxis=dict(
            tickformat='$,.0f',
            gridcolor='#e0e0e0'
        ),
        xaxis=dict(
            gridcolor='#e0e0e0'
        )
    )
    
    return fig

def load_models(mlp_path: str = "outputs/best_mlp.keras", 
                cnn_path: str = "outputs/best_cnn.keras",
                training_script: str = "redes.py") -> tuple:
    """
    Carga los modelos MLP y CNN. Si no existen, ejecuta el script de entrenamiento.
    
    Args:
        mlp_path: Ruta al modelo MLP
        cnn_path: Ruta al modelo CNN
        training_script: Ruta al script de entrenamiento
    
    Returns:
        tuple: (model_mlp, model_cnn) - Modelos de TensorFlow/Keras cargados
    
    Raises:
        RuntimeError: Si el entrenamiento falla
        FileNotFoundError: Si los modelos no se generan después del entrenamiento
    """
    if not os.path.exists(mlp_path) or not os.path.exists(cnn_path):
        print("⚠️  Modelos no encontrados. Ejecutando entrenamiento...")
        print(f"   MLP existe: {os.path.exists(mlp_path)}")
        print(f"   CNN existe: {os.path.exists(cnn_path)}")
        
        try:
            # Ejecutar script de entrenamiento
            result = subprocess.run(
                [sys.executable, training_script],
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
        if not os.path.exists(mlp_path) or not os.path.exists(cnn_path):
            raise FileNotFoundError(
                f"Los modelos no se generaron correctamente después del entrenamiento.\n"
                f"MLP: {os.path.exists(mlp_path)}, CNN: {os.path.exists(cnn_path)}"
            )
    
    # Cargar modelos
    print("📂 Cargando modelos...")
    try:
        model_mlp = tf.keras.models.load_model(mlp_path, compile=False)
        model_cnn = tf.keras.models.load_model(cnn_path, compile=False)
        print("✅ Modelos cargados correctamente")
        return model_mlp, model_cnn
    except Exception as e:
        print(f"❌ Error al cargar los modelos: {e}")
        raise