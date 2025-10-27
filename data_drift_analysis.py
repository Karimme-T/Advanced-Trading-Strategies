# data_drift_analysis.py
"""
Data Drift Modeling (Plotly Focused)
====================================
Analysis focused on the required four points using Plotly:
1. Timeline View: Distribution of features per period (Plotly Histogram + Time Series)
2. Drift Statistics Table: KS-test p-values (Plotly Table)
3. Highlighting: Mark significant drift (Plotly Heatmap)
4. Summary: Top 5 most-drifted features with explanations (Plotly Table & Console)
NO files are saved from this script.
"""
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Asumir que estas variables están disponibles después de Feature_eng.py
try:
    from Feature_eng import train_scaled, val_scaled, test_scaled, feat_cols
except ImportError:
    pass

# ==============================================================================
# 1️⃣ DRIFT STATISTICS TABLE & COMPUTATION (CORE LOGIC)
# ==============================================================================
def compute_drift_statistics():
    """Calcula estadísticas de drift para todas las features usando KS-test."""
    results = []
    
    testable_features = [f for f in feat_cols if pd.api.types.is_numeric_dtype(train_scaled[f])]
    
    for feature in testable_features:
        try:
            train_vals = train_scaled[feature].dropna().values
            val_vals = val_scaled[feature].dropna().values
            ks_tv, p_tv = stats.ks_2samp(train_vals, val_vals)
            
            test_vals = test_scaled[feature].dropna().values
            ks_tt, p_tt = stats.ks_2samp(train_vals, test_vals)
            
            drift_detected = (p_tv < 0.05) or (p_tt < 0.05)
            
            results.append({
                'Feature': feature,
                'KS_Stat_TrainVal': ks_tv,
                'P_Value_TrainVal': p_tv,
                'KS_Stat_TrainTest': ks_tt,
                'P_Value_TrainTest': p_tt,
                'Drift_Detected': drift_detected,
                'Max_KS_Stat': max(ks_tv, ks_tt),
                'Min_P_Value': min(p_tv, p_tt)
            })
        except Exception:
            continue
    
    df_drift = pd.DataFrame(results)
    df_drift = df_drift.sort_values('Max_KS_Stat', ascending=False)
    
    return df_drift.reset_index(drop=True)

def plot_drift_table_plotly(df_drift, top_n=10):
    """Genera tabla interactiva con Plotly mostrando estadísticas de drift (Requisito 2)."""
    df_display = df_drift.head(top_n).copy()
    
    def get_color(drift_detected):
        return '#ffcccc' if drift_detected else '#ccffcc'
    
    cell_colors = [get_color(d) for d in df_display['Drift_Detected']]
    df_display['Status'] = df_display['Drift_Detected'].apply(
        lambda x: '🔴 DRIFT' if x else '🟢 ESTABLE'
    )
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Feature</b>', '<b>Status</b>', '<b>KS Train-Val</b>', '<b>P-Value Train-Val</b>',
                    '<b>KS Train-Test</b>', '<b>P-Value Train-Test</b>'],
            fill_color='#1f77b4', align='left', font=dict(size=12, color='white'), height=35
        ),
        cells=dict(
            values=[
                df_display['Feature'], df_display['Status'], df_display['KS_Stat_TrainVal'].round(4),
                df_display['P_Value_TrainVal'].apply(lambda x: f'{x:.6f}'), df_display['KS_Stat_TrainTest'].round(4),
                df_display['P_Value_TrainTest'].apply(lambda x: f'{x:.6f}')
            ],
            fill_color=[cell_colors] * 6, align='left', font=dict(size=11), height=30
        )
    )])
    
    fig.update_layout(
        title={'text': f'<b>Drift Statistics Table (Top {top_n})</b>', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        height=min(800, 200 + len(df_display) * 35),
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

# ==============================================================================
# 2️⃣ TIMELINE VIEW (REQUISITO 1)
# ==============================================================================
def plot_feature_timeline_plotly(feature):
    """
    Genera Timeline view: distribución sobre los períodos y su evolución temporal (Requisito 1).
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{feature} - Distribution Overlay (Train vs Val vs Test)',
            f'{feature} - Time Series Evolution'
        ),
        specs=[[{"type": "histogram"}], [{"type": "scatter"}]],
        vertical_spacing=0.15
    )
    
    # 1. Histogram/Distribution Overlay
    for data, name, color in [
        (train_scaled[feature].dropna(), 'Train', '#1f77b4'),
        (val_scaled[feature].dropna(), 'Val', '#ff7f0e'),
        (test_scaled[feature].dropna(), 'Test', '#2ca02c')
    ]:
        fig.add_trace(
            go.Histogram(x=data, name=name, marker_color=color, opacity=0.6,
                histnorm='probability density', nbinsx=40, showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Time Series Evolution (with period highlighting)
    if 'date' in train_scaled.columns:
        full_df = pd.concat([train_scaled[['date', feature]], val_scaled[['date', feature]], test_scaled[['date', feature]]]).sort_values('date')
        
        n_train = len(train_scaled)
        n_val = len(val_scaled)
        
        # Train data
        fig.add_trace(go.Scatter(
            x=full_df['date'].iloc[:n_train], y=full_df[feature].iloc[:n_train],
            name='Train', mode='lines', line=dict(color='#1f77b4', width=1), showlegend=False
        ), row=2, col=1)
        # Val data
        fig.add_trace(go.Scatter(
            x=full_df['date'].iloc[n_train:n_train + n_val], y=full_df[feature].iloc[n_train:n_train + n_val],
            name='Validation', mode='lines', line=dict(color='#ff7f0e', width=1.5), showlegend=False
        ), row=2, col=1)
        # Test data
        fig.add_trace(go.Scatter(
            x=full_df['date'].iloc[n_train + n_val:], y=full_df[feature].iloc[n_train + n_val:],
            name='Test', mode='lines', line=dict(color='#2ca02c', width=2), showlegend=False
        ), row=2, col=1)
    
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    fig.update_layout(
        title={'text': f'<b>Timeline View: {feature}</b>', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        height=700, showlegend=True, legend=dict(x=1.05, y=1), margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# ==============================================================================
# 3️⃣ DRIFT HIGHLIGHTING (REQUISITO 3)
# ==============================================================================
def plot_drift_heatmap_plotly(df_drift):
    """Genera Heatmap interactivo con highlighting de p-values (Requisito 3)."""
    drift_matrix = df_drift[['Feature', 'P_Value_TrainVal', 'P_Value_TrainTest']].copy()
    drift_matrix = drift_matrix.set_index('Feature')
    drift_matrix.columns = ['Train vs Val', 'Train vs Test']
    
    hover_text = []
    for i, feat in enumerate(drift_matrix.index):
        row = []
        for j, col in enumerate(drift_matrix.columns):
            p_val = drift_matrix.iloc[i, j]
            drift_status = "🔴 DRIFT" if p_val < 0.05 else "🟢 ESTABLE"
            row.append(f'<b>{feat}</b><br>{col}<br>P-value: {p_val:.6f}<br>Status: {drift_status}')
        hover_text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=drift_matrix.values,
        x=drift_matrix.columns,
        y=drift_matrix.index,
        colorscale=[
            [0.0, '#ff4444'], [0.05, '#ff4444'],      # Rojo (drift: p < 0.05)
            [0.05, '#ffff44'], [0.10, '#ffff44'],     # Amarillo (marginal: 0.05 <= p < 0.10)
            [0.10, '#44ff44'], [1.0, '#44ff44']       # Verde (estable: p >= 0.10)
        ],
        text=drift_matrix.values.round(4),
        texttemplate='%{text}',
        textfont={"size": 9},
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(
            title="P-Value",
            tickvals=[0, 0.05, 0.10, 0.15],
            ticktext=['0.00 (DRIFT)', '0.05 (Threshold)', '0.10', '0.15+ (STABLE)']
        )
    ))
    
    fig.update_layout(
        title={'text': '<b>Data Drift Detection Heatmap</b><br><sub>🔴 Red = Significant Drift (p < 0.05)</sub>',
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        xaxis=dict(title='Period Comparison', side='bottom'),
        yaxis=dict(title='Feature'),
        height=max(600, len(drift_matrix) * 20),
        margin=dict(l=150, r=50, t=120, b=80)
    )
    
    return fig

# ==============================================================================
# 4️⃣ TOP 5 SUMMARY WITH INTERPRETATIONS (REQUISITO 4)
# ==============================================================================
def generate_top5_summary(df_drift):
    """Genera resumen interpretativo de las top 5 features con mayor drift (Requisito 4)."""
    top5 = df_drift.head(5).copy()
    
    interpretations = []
    
    for idx, row in top5.iterrows():
        feature = row['Feature']
        max_ks = row['Max_KS_Stat']
        min_p = row['Min_P_Value']
        
        if min_p < 0.01:
            status = '🔴 CRÍTICO'
        elif min_p < 0.05:
            status = '🟠 ALTO'
        else:
            status = '🟡 MODERADO'
        
        # Interpretación contextual basada en tipo de indicador (simplificada)
        if any(x in feature for x in ['ATR', 'Volatility', 'BB', 'KC', 'Donchian']):
            interpretation = 'Cambio en volatilidad. Posible transición entre régimen de baja/alta volatilidad.'
        elif any(x in feature for x in ['RSI', 'MACD', 'ROC', 'Stoch', 'Momentum', 'Williams', 'AO']):
            interpretation = 'Shift en momentum. Indica cambio de sentimiento alcista/bajista.'
        elif any(x in feature for x in ['Volume', 'OBV', 'VROC', 'MFI', 'CMF', 'AD', 'EOM']):
            interpretation = 'Cambio en patrones de volumen. Puede indicar cambio en liquidez o interés en el activo.'
        elif any(x in feature for x in ['MA', 'EMA', 'ADX', 'PSAR', 'Ichimoku']):
            interpretation = 'Cambio en indicadores de tendencia. Posible reversión o cambio de régimen tendencial.'
        else:
            interpretation = 'Cambio distributivo significativo. Requiere análisis detallado del contexto de mercado.'
        
        magnitude = 'EXTREMA' if max_ks > 0.3 else 'ALTA' if max_ks > 0.15 else 'MODERADA'
        interpretation = f"[Magnitud {magnitude}] {interpretation}"
        
        interpretations.append({
            'Rank': idx + 1,
            'Feature': feature,
            'Max_KS_Stat': max_ks,
            'Min_P_Value': min_p,
            'Drift_Status': status,
            'Interpretation': interpretation
        })
    
    return pd.DataFrame(interpretations)

def plot_top5_summary_plotly(df_summary):
    """Genera tabla interactiva con el resumen de top 5 features (Requisito 4)."""
    
    def get_color(status):
        colors = {'🔴 CRÍTICO': '#ffcccc', '🟠 ALTO': '#ffe6cc', '🟡 MODERADO': '#ffffcc'}
        return colors.get(status, '#ffffff')
    
    cell_colors = [get_color(s) for s in df_summary['Drift_Status']]
    
    fig = go.Figure(data=[go.Table(
        columnwidth=[60, 120, 80, 80, 100, 400],
        header=dict(
            values=['<b>Rank</b>', '<b>Feature</b>', '<b>KS-Stat</b>', '<b>P-Value</b>', '<b>Status</b>', '<b>📝 Interpretation</b>'],
            fill_color='#2ca02c', align='left', font=dict(size=12, color='white'), height=40
        ),
        cells=dict(
            values=[
                df_summary['Rank'], df_summary['Feature'], df_summary['Max_KS_Stat'].round(4),
                df_summary['Min_P_Value'].apply(lambda x: f'{x:.6f}'), df_summary['Drift_Status'],
                df_summary['Interpretation']
            ],
            fill_color=[cell_colors] * 6, align='left', font=dict(size=10), height=70
        )
    )])
    
    fig.update_layout(
        title={'text': '<b>🔍 Top 5 Features with Highest Drift</b><br><sub>Interpretations based on market regime changes</sub>',
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}},
        height=min(700, 250 + len(df_summary) * 70),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    return fig

def print_top5_summary(df_summary):
    """Imprime el resumen de top 5 de forma legible en consola."""
    print("\n" + "="*80)
    print(" "*20 + "🔍 TOP 5 FEATURES CON MAYOR DRIFT 🔍")
    print("="*80)
    
    for _, row in df_summary.iterrows():
        print(f"\n{row['Rank']}. {row['Feature']}")
        print(f"   Status:          {row['Drift_Status']}")
        print(f"   KS-Statistic:    {row['Max_KS_Stat']:.4f}")
        print(f"   P-Value:         {row['Min_P_Value']:.6f}")
        print(f"   📝 Interpretación:")
        print(f"      {row['Interpretation']}")
    
    print("\n" + "="*80)

# ==============================================================================
# 5️⃣ MAIN EXECUTION FUNCTION (FOR PIPELINE)
# ==============================================================================
def run_drift_modeling_analysis(show_plots=True):
    """
    Ejecuta el análisis completo de data drift y genera todos los plots Plotly.
    NO guarda archivos.
    """
    print("\n" + "="*80)
    print(" "*25 + "DATA DRIFT MODELING ANALYSIS")
    print("="*80)
    
    # 1. Compute Drift Statistics
    print("\n📊 Step 1: Computing drift statistics...")
    df_drift = compute_drift_statistics()
    
    total_features = len(df_drift)
    features_with_drift = df_drift['Drift_Detected'].sum()
    
    print(f"   ✓ Analyzed {total_features} features")
    print(f"   ✓ Detected drift in {features_with_drift} features ({features_with_drift/total_features*100:.1f}%)")
    
    # 2. Generate Top 5 Summary
    print("\n🏆 Step 2: Generating Top 5 summary...")
    df_top5 = generate_top5_summary(df_drift)
    print_top5_summary(df_top5)
    
    # 3. Generate and Show Plotly Visualizations (Controlled by main.py)
    if show_plots:
        print("\n📈 Step 3: Generating Plotly visualizations...")
        
        # Table (Requisito 2)
        print("   -> Showing Drift Statistics Table...")
        fig_table = plot_drift_table_plotly(df_drift, top_n=10)
        fig_table.show()
        
        # Heatmap (Requisito 3)
        print("   -> Showing Drift Heatmap (Highlighting)...")
        fig_heatmap = plot_drift_heatmap_plotly(df_drift)
        fig_heatmap.show()
        
        # Top 5 Summary Table (Requisito 4)
        print("   -> Showing Top 5 Summary Table...")
        fig_top5 = plot_top5_summary_plotly(df_top5)
        fig_top5.show()
        # Timeline Views for Top 5 (Requisito 1)
        print("   -> Showing Timeline Views for top 5 features...")
        for _, row in df_top5.iterrows():
            feature = row['Feature']
            print(f"      Plotting: {feature}")
            fig = plot_feature_timeline_plotly(feature)
            fig.show()
    
    return {
        'drift_statistics': df_drift,
        'top5_summary': df_top5
    }
