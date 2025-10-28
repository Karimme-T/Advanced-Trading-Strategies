# Advanced-Trading-Strategies

Sistema completo de trading algorítmico utilizando Deep Learning con validación cruzada temporal y análisis de drift para mercados financieros.

Características Principales

* Modelos Deep Learning: MLP y CNN para predicción de señales de trading
* Cross-Validation Temporal: TimeSeriesSplit para validación robusta sin data leakage
* Data Drift Detection: Monitoreo automático de cambios en distribución de features
* Backtesting: Simulación con stop-loss, take-profit, comisiones y costos de préstamo
* 20+ Indicadores Técnicos: Tendencia, momentum, volatilidad y volumen
* Visualización Interactiva: Dashboards con Plotly para análisis de resultados
* MLflow Integration: Tracking completo de experimentos y métricas

Modelos Implementados
1. Multi-Layer Perceptron (MLP)
   Arquitectura:
- Input: Features normalizadas (20+)
- Hidden Layers: [256, 128] o [512, 256] (configurable)
- Dropout: 0.2-0.4
- Batch Normalization
- Output: 3 clases (Sell=-1, Hold=0, Buy=1)

2. Convolutional Neural Network 1D (CNN)
   Arquitectura:
- Input: Secuencias temporales (lookback=30/60/100)
- Conv1D Layers: [128, 256] filters
- MaxPooling + GlobalAveragePooling
- Dropout: 0.3-0.4
- Output: 3 clases (Sell=-1, Hold=0, Buy=1)

Visualizaciones
El sistema genera automáticamente:
1. Curva de Equity

Evolución del capital en el tiempo
Comparación MLP vs CNN vs Buy & Hold

2. Métricas Comparativas

Dashboard interactivo con Plotly
Sharpe, Sortino, Calmar ratios
Drawdown y win rate

3. Distribución de Trades

Histograma de P&L por trade
Análisis de wins vs losses

4. Matrices de Confusión

Performance de clasificación
Por modelo y por fold

5. Drift Analysis

Heatmap de features con drift
KS-test p-values
Top features afectadas

Experimentos con MLflow
Visualizar Experimentos
mlflow ui

Abre en navegador: http://localhost:5000
Experimentos Registrados

dl_trading: Modelos sin CV
dl_trading_cv: Modelos con CV (recomendado)

Métricas Tracked

Accuracy, F1-score, AUC
Métricas de CV (mean ± std)
Hiperparámetros
Curvas de entrenamiento
Matrices de confusión
Métricas de backtesting


Disclaimer
Este software es solo para propósitos educativos e investigación. NO ES CONSEJO FINANCIERO.

El trading conlleva riesgos significativos
El rendimiento pasado no garantiza resultados futuros
Usa capital que puedas permitirte perder
Consulta con un asesor financiero antes de invertir
Los autores no son responsables por pérdidas financieras


Licencia
Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.
