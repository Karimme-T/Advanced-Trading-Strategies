import os, json, numpy as np
import mlflow, mlflow.tensorflow
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from Feature_eng import make_sequences, train_scaled, val_scaled, test_scaled, feat_cols

experiment_name = "dl_trading"
lookback = 30         
epochs = 50
batch_size = 128
patience = 8
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)


def remap_labels(y_raw: np.ndarray):
    """Mapea {-1,0,1} -> {0,1,2} (ordenado) y devuelve (y_int, mapping_dict)."""
    clases = sorted(np.unique(y_raw))
    mapping = {lab:i for i, lab in enumerate(clases)}
    y_int = np.vectorize(mapping.get)(y_raw).astype(int)
    return y_int, mapping

def inv_map_from_mapping(mapping: dict):
    return {v:k for k,v in mapping.items()}

def class_weights_balanced(y_int: np.ndarray) -> dict:
    clases = np.unique(y_int)
    w = compute_class_weight(class_weight="balanced", classes=clases, y=y_int)
    return {int(c): float(wi) for c, wi in zip(clases, w)}

def xy_daywise(df_part, feat_cols):
    X = df_part[feat_cols].to_numpy(dtype=np.float32)
    y_raw = df_part["signal"].to_numpy()
    y, mapping = remap_labels(y_raw)
    return X, y, mapping

def binarize_for_auc(y_true_int: np.ndarray, n_classes: int) -> np.ndarray:
    return label_binarize(y_true_int, classes=list(range(n_classes)))

def save_cm_and_report(y_true, y_pred, split_name: str):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Matriz de confusión — {split_name}")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    cm_path = os.path.join(outdir, f"cm_{split_name}.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    rep_txt = classification_report(y_true, y_pred)
    rep_path = os.path.join(outdir, f"clf_report_{split_name}.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_txt)
    return cm_path, rep_path

def evaluate_split(y_true_int: np.ndarray, proba: np.ndarray, split: str):
    y_pred_int = proba.argmax(axis=1)
    acc = accuracy_score(y_true_int, y_pred_int)
    f1 = f1_score(y_true_int, y_pred_int, average="macro")
    try:
        auc = roc_auc_score(binarize_for_auc(y_true_int, proba.shape[1]),
                            proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")
    cm_path, rep_path = save_cm_and_report(y_true_int, y_pred_int, split)
    mlflow.log_metrics({f"{split}_accuracy": acc, f"{split}_f1_macro": f1, f"{split}_auc_macro_ovr": auc})
    mlflow.log_artifact(cm_path); mlflow.log_artifact(rep_path)
    return {"accuracy": acc, "f1_macro": f1, "auc_macro_ovr": auc}


#Modelos
def build_mlp(input_dim: int, num_classes: int = 3) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_cnn(input_len: int, num_features: int, num_classes: int = 3,
              kernel_size: int = 5, pool_size: int = 2, filters: int = 64) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, num_features))
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    x = layers.Conv1D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# Construcción de datasets X, y

# Para MLP (día a día)
X_tr_mlp, y_tr_mlp, map_mlp = xy_daywise(train_scaled, feat_cols)
X_va_mlp, y_va_mlp, _        = xy_daywise(val_scaled,   feat_cols)
X_te_mlp, y_te_mlp, _        = xy_daywise(test_scaled,  feat_cols)
n_clases = len(np.unique(np.r_[y_tr_mlp, y_va_mlp, y_te_mlp]))

# Para CNN 
X_tr_seq, y_tr_seq_raw = make_sequences(train_scaled, feat_cols, lookback)
X_va_seq, y_va_seq_raw = make_sequences(val_scaled,   feat_cols, lookback)
X_te_seq, y_te_seq_raw = make_sequences(test_scaled,  feat_cols, lookback)

# Remapeo coherente para CNN 
y_tr_seq = np.vectorize(map_mlp.get)(y_tr_seq_raw).astype(int)
y_va_seq = np.vectorize(map_mlp.get)(y_va_seq_raw).astype(int)
y_te_seq = np.vectorize(map_mlp.get)(y_te_seq_raw).astype(int)


# Entrenamiento y comparación
mlflow.set_experiment(experiment_name)
try:
    import mlflow.keras
    mlflow.keras.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True,
        log_datasets=False
    )
except Exception:
    import mlflow.tensorflow
    mlflow.tensorflow.autolog(
        log_models=True,
        log_input_examples=False,
        log_model_signatures=True
    )

resultados_val = {}

#  MLP
with mlflow.start_run(run_name="MLP_baseline") as run:
    mlflow.log_params({
        "model": "MLP",
        "n_features": X_tr_mlp.shape[1],
        "epochs": epochs, "batch_size": batch_size, "patience": patience
    })
    pesos = class_weights_balanced(y_tr_mlp)
    modelo_mlp = build_mlp(input_dim=X_tr_mlp.shape[1], num_classes=n_clases)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(os.path.join(outdir, "best_mlp"),
                                     monitor="val_loss", save_best_only=True, save_weights_only=False)
    hist = modelo_mlp.fit(X_tr_mlp, y_tr_mlp,
                          validation_data=(X_va_mlp, y_va_mlp),
                          epochs=epochs, batch_size=batch_size,
                          class_weight=pesos,
                          callbacks=[es, ckpt], verbose=2)
    modelo_mlp.save(os.path.join(outdir, "best_mlp"), include_optimizer=False)

    # Evaluación
    proba_val = modelo_mlp.predict(X_va_mlp)
    proba_te  = modelo_mlp.predict(X_te_mlp)
    m_val = evaluate_split(y_va_mlp, proba_val, "val_mlp")
    m_te  = evaluate_split(y_te_mlp, proba_te,  "test_mlp")
    resultados_val["MLP"] = m_val
    with open(os.path.join(outdir, "mlp_test_metrics.json"), "w") as f:
        json.dump(m_te, f, indent=2)
    mlflow.log_artifact(os.path.join(outdir, "mlp_test_metrics.json"))

# CNN 1D 
with mlflow.start_run(run_name="CNN_1D") as run:
    mlflow.log_params({
        "model": "CNN1D",
        "lookback": lookback,
        "n_features": X_tr_seq.shape[2],
        "epochs": epochs, "batch_size": batch_size, "patience": patience
    })
    pesos = class_weights_balanced(y_tr_seq)
    modelo_cnn = build_cnn(input_len=X_tr_seq.shape[1], num_features=X_tr_seq.shape[2],
                           num_classes=n_clases)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ckpt = callbacks.ModelCheckpoint(os.path.join(outdir, "best_cnn"),
                                     monitor="val_loss", save_best_only=True, save_weights_only=False)
    hist = modelo_cnn.fit(X_tr_seq, y_tr_seq,
                          validation_data=(X_va_seq, y_va_seq),
                          epochs=epochs, batch_size=batch_size,
                          class_weight=pesos,
                          callbacks=[es, ckpt], verbose=2)
    modelo_cnn.save(os.path.join(outdir, "best_cnn"), include_optimize=False)
    
    # Evaluación
    proba_val = modelo_cnn.predict(X_va_seq)
    proba_te  = modelo_cnn.predict(X_te_seq)
    m_val = evaluate_split(y_va_seq, proba_val, "val_cnn")
    m_te  = evaluate_split(y_te_seq,  proba_te,  "test_cnn")
    resultados_val["CNN"] = m_val
    with open(os.path.join(outdir, "cnn_test_metrics.json"), "w") as f:
        json.dump(m_te, f, indent=2)
    mlflow.log_artifact(os.path.join(outdir, "cnn_test_metrics.json"))

# Selección del mejor por F1 en VALID 
mejor = max(resultados_val.items(), key=lambda kv: kv[1]["f1_macro"])[0]
print(f"Mejor modelo (VALID F1): {mejor}")
with mlflow.start_run(run_name=f"BEST_{mejor}_TEST") as run:
    mlflow.log_param("best_model", mejor)
