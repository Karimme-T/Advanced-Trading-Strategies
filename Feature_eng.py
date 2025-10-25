# Importaciones
import numpy as np
import pandas as pd
import datetime
import ta
from sklearn.preprocessing import StandardScaler
import dateparser

# Carga de datos
df = pd.read_csv('amzn_data.csv')
df.head()

# 20 Indicadores técnicos
def technical_indicators(df):
    """
    Adds trend, momentum, volatility and volume indicators to the DataFrame.
    Input: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
    Output: New DataFrame with additional columns for each indicator
    """
    data = df.copy()

    # ========== TENDENCIA ==========
    data['MA_20'] = ta.trend.SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()

    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()

    adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
    data['ADX'] = adx.adx()

    psar = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close'])
    data['Parabolic_SAR'] = psar.psar()

    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'])
    data['Ichimoku_base'] = ichimoku.ichimoku_base_line()
    data['Ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

    # ========== MOMENTUM ==========
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data['ROC'] = ta.momentum.ROCIndicator(data['Close']).roc()
    data['Stoch'] = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()
    data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
    data['Momentum'] = data['Close'].diff(10)
    data['Williams_%R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
    data['AO'] = ta.momentum.AwesomeOscillatorIndicator(data['High'], data['Low']).awesome_oscillator()

    # ========== VOLATILIDAD ==========
    data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()

    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_high'] = bb.bollinger_hband()
    data['BB_low'] = bb.bollinger_lband()

    kc = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'])
    data['KC_high'] = kc.keltner_channel_hband()
    data['KC_low'] = kc.keltner_channel_lband()

    dc = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'])
    data['Donchian_high'] = dc.donchian_channel_hband()
    data['Donchian_low'] = dc.donchian_channel_lband()

    n = 10  # periodo
    hl_range = data['High'] - data['Low']
    data['Chaikin_Volatility'] = (
        (hl_range.ewm(span=n).mean() - hl_range.ewm(span=n).mean().shift(n))
        / hl_range.ewm(span=n).mean().shift(n)
    ) * 100

    # ========== VOLUMEN ==========
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
    data['VROC'] = data['Volume'].pct_change(10)
    data['MFI'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
    data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume']).chaikin_money_flow()
    data['AD'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
    data['EOM'] = ta.volume.EaseOfMovementIndicator(data['High'], data['Low'], data['Volume']).ease_of_movement()

    # ========== OTROS ==========
    data['Pivot_Point'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()
    data['ATR_Bands_high'] = data['Close'] + 2 * data['ATR']
    data['ATR_Bands_low'] = data['Close'] - 2 * data['ATR']

    # SuperTrend (custom calc, since ta doesn’t have built-in)
    multiplier = 3
    atr = data['ATR']
    hl2 = (data['High'] + data['Low']) / 2
    data['SuperTrend'] = hl2 - (multiplier * atr)

    # Ulcer Index (custom calc)
    drawdown = (data['Close'] / data['Close'].cummax() - 1) * 100
    data['Ulcer_Index'] = (drawdown ** 2).rolling(14).mean() ** 0.5

    return data


# Aplicando en data
data_ind = technical_indicators(df)

# Preproceso
def preprocess(df):
    # Renombrar la columna de fecha
    if 'Price' in df.columns:
        df = df.rename(columns={'Price': 'date'})
    
    # Convertir la columna 'date' a formato datetime usando dateparser
    df['date'] = df['date'].apply(lambda x: dateparser.parse(str(x)))
    
    # Eliminar filas con fechas inválidas
    df = df.dropna(subset=['date'])
    
    # Ordenar de más antiguo a más reciente
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
    
    # Eliminar datos nulos en el resto de columnas
    df = df.dropna().reset_index(drop=True)
    
    # Hacer la fecha el índice temporal para estandarizar
    df = df.set_index('date')
    
    # Calcular los índices de split
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    # Separar en train, validation y test
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Calcular medias y desviaciones del train
    mean = train_df.mean()
    std = train_df.std(ddof=0)
    
    # Estandarizar con base a estadísticas del train
    train_scaled = (train_df - mean) / std
    val_scaled = (val_df - mean) / std
    test_scaled = (test_df - mean) / std
    
    # Volver a colocar 'date' como columna
    train_scaled = train_scaled.reset_index()
    val_scaled = val_scaled.reset_index()
    test_scaled = test_scaled.reset_index()
    
    # Devolver los tres datasets
    return train_scaled, val_scaled, test_scaled

train_scaled, val_scaled, test_scaled = preprocess(data_ind)

# Señales de respuesta
def set_response_var(train_scaled, threshold=0.015):
    """
    Crea una variable de respuesta 'signal' para trading basada en los cambios del precio estandarizado 'Close'.
    Args:
        train_scaled (pd.DataFrame): DataFrame con la columna 'Close' estandarizada.
        threshold (float): Umbral mínimo de cambio absoluto para considerar 'hold'. Por defecto 0.0.
    Returns:
        pd.DataFrame: train_scaled con una nueva columna 'signal' que toma valores {-1, 0, 1}.
    """
    
    # Verificar que exista la columna 'Close'
    if 'Close' not in train_scaled.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close' estandarizada.")
    
    df = train_scaled.copy()
    
    # Calcular el cambio día a día de 'Close'
    df['close_diff'] = df['Close'].diff()
    
    # Aplicar la lógica de señales
    df['signal'] = np.where(df['close_diff'] > threshold, 1,     # Subió → Buy
                    np.where(df['close_diff'] < -threshold, -1,  # Bajó → Sell
                    0))                                           # Igual → Hold
    
    # Eliminar la columna auxiliar
    df = df.drop(columns=['close_diff'])
    
    return df

train_df = set_response_var(train_scaled)

train_df['signal'].value_counts()