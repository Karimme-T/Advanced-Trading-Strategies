# data_drift_analysis.py
"""
Data Drift Analysis - Detección de cambios en distribuciones de features
Sin dependencias de Evidently - Solo scipy, pandas, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from Feature_eng import train_scaled, val_scaled, test_scaled, feat_cols
import os

OUTPUT_DIR = "outputs/drift_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def ks_test_drift(train_data, test_data, feature):
    """
    Kolmogorov-Smirnov test para detectar drift
    H0: Las distribuciones son iguales
    p-value < 0.05 → rechazamos H0 → hay drift
    
    Parámetros:
    -----------
    train_data : pd.DataFrame
        Datos de entrenamiento
    test_data : pd.DataFrame
        Datos de validación/test
    feature : str
        Nombre de la feature a analizar
    
    Retorna:
    --------
    tuple : (statistic, p_value)
    """
    # Remover NaN antes del test
    train_values = train_data[feature].dropna()
    test_values = test_data[feature].dropna()
    
    statistic, p_value = stats.ks_2samp(train_values, test_values)
    return statistic, p_value


def chi_squared_test(train_data, test_data, feature, bins=10):
    """
    Chi-squared test para variables categóricas/discretas
    
    Parámetros:
    -----------
    train_data : pd.DataFrame
        Datos de entrenamiento
    test_data : pd.DataFrame
        Datos de test
    feature : str
        Nombre de la feature
    bins : int
        Número de bins para discretizar
    
    Retorna:
    --------
    tuple : (chi2, p_value)
    """
    train_values = train_data[feature].dropna()
    test_values = test_data[feature].dropna()
    
    # Crear bins comunes
    train_hist, bin_edges = np.histogram(train_values, bins=bins)
    test_hist, _ = np.histogram(test_values, bins=bin_edges)
    
    # Evitar divisiones por cero
    train_hist = train_hist + 1
    test_hist = test_hist + 1
    
    chi2, p_value = stats.chisquare(test_hist, train_hist)
    return chi2, p_value


def analyze_drift_all_features():
    """
    Analiza drift para todas las features entre train/val/test
    
    Retorna:
    --------
    pd.DataFrame : Tabla con estadísticas de drift por feature
    """
    results = []
    
    for feature in feat_cols:
        try:
            # Train vs Val
            ks_stat_tv, p_val_tv = ks_test_drift(train_scaled, val_scaled, feature)
            
            # Train vs Test
            ks_stat_tt, p_val_tt = ks_test_drift(train_scaled, test_scaled, feature)
            
            # Val vs Test
            ks_stat_vt, p_val_vt = ks_test_drift(val_scaled, test_scaled, feature)
            
            # Determinar si hay drift (p < 0.05)
            drift_tv = p_val_tv < 0.05
            drift_tt = p_val_tt < 0.05
            drift_vt = p_val_vt < 0.05
            
            results.append({
                'feature': feature,
                'ks_stat_train_val': ks_stat_tv,
                'p_value_train_val': p_val_tv,
                'drift_train_val': drift_tv,
                'ks_stat_train_test': ks_stat_tt,
                'p_value_train_test': p_val_tt,
                'drift_train_test': drift_tt,
                'ks_stat_val_test': ks_stat_vt,
                'p_value_val_test': p_val_vt,
                'drift_val_test': drift_vt,
                'max_ks_stat': max(ks_stat_tv, ks_stat_tt, ks_stat_vt),
                'min_p_value': min(p_val_tv, p_val_tt, p_val_vt)
            })
        except Exception as e:
            print(f"   ⚠️  Error al analizar {feature}: {e}")
            continue
    
    df_drift = pd.DataFrame(results)
    df_drift = df_drift.sort_values('max_ks_stat', ascending=False)
    
    return df_drift


def plot_drift_timeline(feature, save_path=None):
    """
    Plotea distribución de un feature a través del tiempo
    
    Parámetros:
    -----------
    feature : str
        Nombre de la feature a plotear
    save_path : str, optional
        Ruta donde guardar la imagen
    
    Retorna:
    --------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogramas por período
    axes[0, 0].hist(train_scaled[feature].dropna(), bins=50, alpha=0.7, 
                    label='Train', color='blue', density=True)
    axes[0, 0].hist(val_scaled[feature].dropna(), bins=50, alpha=0.7, 
                    label='Val', color='orange', density=True)
    axes[0, 0].hist(test_scaled[feature].dropna(), bins=50, alpha=0.7, 
                    label='Test', color='green', density=True)
    axes[0, 0].set_title(f'{feature} - Histogram Overlay')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # KDE plots
    train_scaled[feature].dropna().plot(kind='kde', ax=axes[0, 1], 
                                        label='Train', color='blue', linewidth=2)
    val_scaled[feature].dropna().plot(kind='kde', ax=axes[0, 1], 
                                      label='Val', color='orange', linewidth=2)
    test_scaled[feature].dropna().plot(kind='kde', ax=axes[0, 1], 
                                       label='Test', color='green', linewidth=2)
    axes[0, 1].set_title(f'{feature} - KDE')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Box plots
    data_to_plot = [
        train_scaled[feature].dropna(),
        val_scaled[feature].dropna(),
        test_scaled[feature].dropna()
    ]
    axes[1, 0].boxplot(data_to_plot, labels=['Train', 'Val', 'Test'])
    axes[1, 0].set_title(f'{feature} - Box Plot')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(alpha=0.3)
    
    # Time series (si hay fecha)
    if 'date' in train_scaled.columns:
        train_dates = train_scaled['date'].reset_index(drop=True)
        val_dates = val_scaled['date'].reset_index(drop=True)
        test_dates = test_scaled['date'].reset_index(drop=True)
        
        train_vals = train_scaled[feature].reset_index(drop=True)
        val_vals = val_scaled[feature].reset_index(drop=True)
        test_vals = test_scaled[feature].reset_index(drop=True)
        
        axes[1, 1].plot(train_dates, train_vals, label='Train', 
                       alpha=0.7, linewidth=0.5, color='blue')
        axes[1, 1].plot(val_dates, val_vals, label='Val', 
                       alpha=0.7, linewidth=0.5, color='orange')
        axes[1, 1].plot(test_dates, test_vals, label='Test', 
                       alpha=0.7, linewidth=0.5, color='green')
        axes[1, 1].set_title(f'{feature} - Time Series')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No date column available', 
                       ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_drift_heatmap(df_drift):
    """
    Heatmap de p-values para visualizar drift
    
    Parámetros:
    -----------
    df_drift : pd.DataFrame
        DataFrame con resultados de drift analysis
    
    Retorna:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, max(8, len(df_drift) * 0.3)))
    
    # Preparar matriz de p-values
    drift_matrix = df_drift[['feature', 'p_value_train_val', 
                             'p_value_train_test', 'p_value_val_test']].set_index('feature')
    drift_matrix.columns = ['Train vs Val', 'Train vs Test', 'Val vs Test']
    
    # Heatmap
    sns.heatmap(
        drift_matrix,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',
        center=0.05,
        vmin=0,
        vmax=0.15,
        cbar_kws={'label': 'p-value'},
        ax=ax
    )
    
    ax.set_title('Data Drift Detection (KS-Test p-values)\nRed = Significant Drift (p < 0.05)', 
                fontsize=14, pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'drift_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig


def generate_drift_report():
    """
    Genera reporte completo de drift analysis
    
    Retorna:
    --------
    pd.DataFrame : Tabla completa con estadísticas de drift
    """
    print("="*60)
    print("DATA DRIFT ANALYSIS")
    print("="*60)
    
    # 1. Analizar todas las features
    print("\n1. Analizando drift en todas las features...")
    df_drift = analyze_drift_all_features()
    
    # Guardar tabla completa
    csv_path = os.path.join(OUTPUT_DIR, 'drift_statistics.csv')
    df_drift.to_csv(csv_path, index=False)
    print(f"   ✓ Tabla guardada: {csv_path}")
    
    # 2. Top 5 features con más drift
    print("\n2. Top 5 Features con Mayor Drift:")
    print("-" * 60)
    top5 = df_drift.head(5)
    for idx, row in top5.iterrows():
        print(f"\n   {row['feature']}")
        print(f"      KS-statistic (max): {row['max_ks_stat']:.4f}")
        print(f"      p-value (min):      {row['min_p_value']:.6f}")
        print(f"      Drift Train-Val:    {'YES' if row['drift_train_val'] else 'NO'}")
        print(f"      Drift Train-Test:   {'YES' if row['drift_train_test'] else 'NO'}")
        print(f"      Drift Val-Test:     {'YES' if row['drift_val_test'] else 'NO'}")
    
    # 3. Resumen estadístico
    print("\n3. Resumen Estadístico:")
    print("-" * 60)
    total_features = len(df_drift)
    drift_train_val = df_drift['drift_train_val'].sum()
    drift_train_test = df_drift['drift_train_test'].sum()
    drift_val_test = df_drift['drift_val_test'].sum()
    
    print(f"   Total features analizados: {total_features}")
    print(f"   Features con drift Train-Val:  {drift_train_val} ({drift_train_val/total_features*100:.1f}%)")
    print(f"   Features con drift Train-Test: {drift_train_test} ({drift_train_test/total_features*100:.1f}%)")
    print(f"   Features con drift Val-Test:   {drift_val_test} ({drift_val_test/total_features*100:.1f}%)")
    
    # 4. Generar heatmap
    print("\n4. Generando heatmap de drift...")
    plot_drift_heatmap(df_drift)
    print(f"   ✓ Heatmap guardado: {OUTPUT_DIR}/drift_heatmap.png")
    
    # 5. Timeline plots para top 5
    print("\n5. Generando timeline plots para top 5 features...")
    for idx, row in top5.iterrows():
        feature = row['feature']
        save_path = os.path.join(OUTPUT_DIR, f'timeline_{feature}.png')
        try:
            plot_drift_timeline(feature, save_path)
            print(f"   ✓ {feature}")
        except Exception as e:
            print(f"   ⚠️  Error al plotear {feature}: {e}")
    
    # 6. Interpretación
    print("\n6. Interpretación de Market Regimes:")
    print("-" * 60)
    print("""
    Las features con mayor drift pueden indicar:
    
    • Cambios en volatilidad del mercado (ATR, Bollinger Bands)
    • Shifts en momentum (RSI, MACD, ROC)
    • Cambios en volumen de trading (OBV, MFI, VROC)
    • Transiciones entre regímenes alcistas/bajistas
    • Eventos de mercado (crisis, rallies, consolidaciones)
    
    Features con drift significativo requieren:
    → Reentrenamiento periódico del modelo
    → Monitoring continuo de performance
    → Ajuste de umbrales CFA según régimen
    """)
    
    print("\n" + "="*60)
    print("DRIFT ANALYSIS COMPLETADO")
    print(f"Archivos generados en: {OUTPUT_DIR}/")
    print("="*60)
    
    return df_drift


