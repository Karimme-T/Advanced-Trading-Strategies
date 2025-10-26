# Importaciones
import numpy as np
import pandas as pd
import datetime
import ta
from sklearn.preprocessing import StandardScaler
import dateparser


# Parámetros 
umbral_alcista = 0.014       
umbral_bajista = -0.014     
horizonte = 1         
train_f = 0.60
val_f = 0.20

df = pd.read_csv("amzn_data.csv")

if "Price" not in df.columns:
    raise ValueError("No esta la columna 'Price' (fecha) en el CSV.")

df = df.rename(columns={"Price": "date"})
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


# Indicadores técnicos 
def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds trend, momentum, volatility and volume indicators to the DataFrame.
    Input: DataFrame with columns ['Open','High','Low','Close','Volume']
    Output: DataFrame with additional indicator columns (sin NaNs de warm-up).
    """
    data = df.copy()

    # Tendencia 
    data["MA_20"]  = ta.trend.SMAIndicator(data["Close"], window=20).sma_indicator()
    data["EMA_20"] = ta.trend.EMAIndicator(data["Close"], window=20).ema_indicator()

    macd = ta.trend.MACD(data["Close"])
    data["MACD"]        = macd.macd()
    data["MACD_signal"] = macd.macd_signal()

    adx = ta.trend.ADXIndicator(data["High"], data["Low"], data["Close"], window=14)
    data["ADX"] = adx.adx()

    psar = ta.trend.PSARIndicator(data["High"], data["Low"], data["Close"])
    data["Parabolic_SAR"] = psar.psar()

    ichi = ta.trend.IchimokuIndicator(data["High"], data["Low"], window1=9, window2=26, window3=52)
    data["Ichimoku_base"]       = ichi.ichimoku_base_line()
    data["Ichimoku_conversion"] = ichi.ichimoku_conversion_line()

    # Momentum
    data["RSI"]      = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["ROC"]      = ta.momentum.ROCIndicator(data["Close"], window=12).roc()
    data["Stoch"]    = ta.momentum.StochasticOscillator(data["High"], data["Low"], data["Close"]).stoch()
    data["CCI"]      = ta.trend.CCIIndicator(data["High"], data["Low"], data["Close"], window=20).cci()
    data["Momentum"] = data["Close"].diff(10)
    data["Williams_%R"] = ta.momentum.WilliamsRIndicator(
        data["High"], data["Low"], data["Close"], lbp=14
    ).williams_r()
    data["AO"] = ta.momentum.AwesomeOscillatorIndicator(data["High"], data["Low"]).awesome_oscillator()

    # Volatilidad
    data["ATR"] = ta.volatility.AverageTrueRange(
        data["High"], data["Low"], data["Close"], window=14
    ).average_true_range()

    bb = ta.volatility.BollingerBands(data["Close"], window=20, window_dev=2)
    data["BB_high"] = bb.bollinger_hband()
    data["BB_low"]  = bb.bollinger_lband()

    kc = ta.volatility.KeltnerChannel(data["High"], data["Low"], data["Close"], window=20)
    data["KC_high"] = kc.keltner_channel_hband()
    data["KC_low"]  = kc.keltner_channel_lband()

    dc = ta.volatility.DonchianChannel(data["High"], data["Low"], data["Close"], window=20)
    data["Donchian_high"] = dc.donchian_channel_hband()
    data["Donchian_low"]  = dc.donchian_channel_lband()

    n = 10
    hl_range = data["High"] - data["Low"]
    ema_hl   = hl_range.ewm(span=n).mean()
    data["Chaikin_Volatility"] = ((ema_hl - ema_hl.shift(n)) / ema_hl.shift(n)) * 100

    # Volumen
    data["OBV"]  = ta.volume.OnBalanceVolumeIndicator(data["Close"], data["Volume"]).on_balance_volume()
    data["VROC"] = data["Volume"].pct_change(10)
    data["MFI"]  = ta.volume.MFIIndicator(
        data["High"], data["Low"], data["Close"], data["Volume"], window=14
    ).money_flow_index()
    data["CMF"]  = ta.volume.ChaikinMoneyFlowIndicator(
        data["High"], data["Low"], data["Close"], data["Volume"], window=20
    ).chaikin_money_flow()
    data["AD"]   = ta.volume.AccDistIndexIndicator(data["High"], data["Low"], data["Close"], data["Volume"]).acc_dist_index()
    data["EOM"]  = ta.volume.EaseOfMovementIndicator(data["High"], data["Low"], data["Volume"], window=14).ease_of_movement()

    # Otros 
    data["Pivot_Point"]    = (data["High"] + data["Low"] + data["Close"]) / 3
    data["VWAP"]           = ta.volume.VolumeWeightedAveragePrice(
        data["High"], data["Low"], data["Close"], data["Volume"], window=14
    ).volume_weighted_average_price()
    data["ATR_Bands_high"] = data["Close"] + 2 * data["ATR"]
    data["ATR_Bands_low"]  = data["Close"] - 2 * data["ATR"]

    multiplier = 3
    hl2 = (data["High"] + data["Low"]) / 2
    data["SuperTrend"] = hl2 - (multiplier * data["ATR"])

    # Ulcer Index 
    drawdown = (data["Close"] / data["Close"].cummax() - 1) * 100
    data["Ulcer_Index"] = (drawdown.pow(2).rolling(14).mean()).pow(0.5)

    data = data.dropna().reset_index(drop=True)
    return data

data_ind = technical_indicators(df)

# Señal con RETORNO FUTURO (sin fuga, con Close real)
def add_future_signal(d: pd.DataFrame,
                      up_th: float = umbral_alcista,
                      down_th: float = umbral_bajista,
                      horizon: int = horizonte,
                      price_col: str = "Close") -> pd.DataFrame:
    d = d.copy()
    if price_col not in d.columns:
        raise ValueError(f"Falta la columna '{price_col}'.")
    
    d["ret_1d_fut"] = d[price_col].shift(-horizon) / d[price_col] - 1.0
    d["signal"] = np.select(
        [d["ret_1d_fut"] >= up_th, d["ret_1d_fut"] <= down_th],
        [1, -1],
        default=0
    )

    d = d.dropna(subset=["ret_1d_fut"]).reset_index(drop=True)
    return d

data_with_y = add_future_signal(data_ind, umbral_alcista, umbral_bajista, horizonte, "Close")

# Split temporal + escalado de features 
def split_and_scale(d: pd.DataFrame,
                    train_frac: float = train_f,
                    val_frac: float = val_f):
    
    assert np.isclose(train_frac + val_frac, 0.8)
    d = d.sort_values("date").reset_index(drop=True)

    n = len(d)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))

    train = d.iloc[:i_tr].copy()
    val   = d.iloc[i_tr:i_va].copy()
    test  = d.iloc[i_va:].copy()

    exclude = {"date", "signal", "ret_1d_fut"}
    feat_cols = [c for c in d.columns if c not in exclude and pd.api.types.is_numeric_dtype(d[c])]

    mean = train[feat_cols].mean()
    std  = train[feat_cols].std(ddof=0).replace(0, 1.0)

    def scale(part):
        out = part.copy()
        out[feat_cols] = (out[feat_cols] - mean) / std
        return out

    return scale(train), scale(val), scale(test), feat_cols, mean, std

train_scaled, val_scaled, test_scaled, feat_cols, mean, std = split_and_scale(data_with_y)

print("Distribución de clases:")
print("train:", train_scaled["signal"].value_counts(normalize=True).round(3))
print("val  :", val_scaled["signal"].value_counts(normalize=True).round(3))
print("test :", test_scaled["signal"].value_counts(normalize=True).round(3))

# Empaquetar ventanas para Keras
def make_sequences(df_part: pd.DataFrame, feat_cols: list, lookback: int = 30):
    """
    Convierte un dataframe ya escalado en tensores (X, y) con ventana deslizante.
    Alineación: X[t-lookback+1 ... t] -> y[t] (usa 'signal' en el mismo índice).
    NOTA: df_part debe estar ordenado por 'date'.
    """
    d = df_part.sort_values("date").reset_index(drop=True)
    X_list, y_list = [], []
    for t in range(lookback - 1, len(d)):
        X_win = d.loc[t - lookback + 1:t, feat_cols].to_numpy(dtype=np.float32)
        y_t   = d.loc[t, "signal"]
        X_list.append(X_win)
        y_list.append(y_t)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y

