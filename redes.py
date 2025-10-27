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
lookback = 10       
epochs = 200
batch_size = 256
patience = 15
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)

def make_sparse_ce(label_smoothing: float = 0.0):
    try:
        # Algunas versiones sí lo soportan:
        return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing and label_smoothing > 0:
            print("[WARN] label_smoothing no soportado en esta versión de TF con SparseCategoricalCrossentropy; usando loss sin smoothing.")
        return "sparse_categorical_crossentropy"



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
def build_mlp(input_dim: int, num_classes: int = 3, params: dict | None = None) -> tf.keras.Model:
    params = params or {}
    # Hiperparámetros con defaults
    hidden = params.get("hidden", [256, 128])
    drop = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 1e-3))
    act = params.get("activation", "relu")
    l2w = float(params.get("l2", 1e-4))
    label_smoothing = float(params.get("label_smoothing", 0.0))

    reg = tf.keras.regularizers.l2(l2w)
    model = models.Sequential([layers.Input(shape=(input_dim,))])
    for h in hidden:
        model.add(layers.Dense(int(h), activation=act, kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(drop))
    model.add(layers.Dense(num_classes, activation="softmax"))

    loss_fn = make_sparse_ce(label_smoothing=label_smoothing)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss = loss_fn,
        metrics = ["accuracy"]
    )
    return model


def build_cnn(input_len: int, num_features: int, num_classes: int = 3, params: dict | None = None) -> tf.keras.Model:
    params = params or {}
    # Hiperparámetros con defaults
    f1 = int(params.get("filters1", 128))
    f2 = int(params.get("filters2", 256))
    k1 = int(params.get("kernel1", 5))
    k2 = int(params.get("kernel2", 3))
    pool = int(params.get("pool", 2))
    drop = float(params.get("dropout", 0.3))
    lr = float(params.get("lr", 1e-3))
    l2w = float(params.get("l2", 1e-4))
    act = params.get("activation", "relu")
    label_smoothing = float(params.get("label_smoothing", 0.0))

    reg = tf.keras.regularizers.l2(l2w)
    inp = layers.Input(shape=(input_len, num_features))
    x = layers.Conv1D(f1, kernel_size=k1, padding="same", activation=act, kernel_regularizer=reg)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=pool)(x)
    x = layers.Conv1D(f2, kernel_size=k2, padding="same", activation=act, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(256, activation=act, kernel_regularizer=reg)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)


    loss_fn = make_sparse_ce(label_smoothing=label_smoothing)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss = loss_fn,
        metrics = ["accuracy"]
    )
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

mlp_space = [
    {"model": "MLP", "hidden": [256,128], "dropout": 0.2, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.0},
    {"model": "MLP", "hidden": [512,256], "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
    {"model": "MLP", "hidden": [256,256,128], "dropout": 0.3, "lr": 5e-4, "activation": "relu", "l2": 5e-4, "label_smoothing": 0.05},
]

cnn_space = [
    # Variamos lookback → se regeneran secuencias por cada HP set
    {"model": "CNN1D", "lookback": 30, "filters1": 128, "filters2": 256, "kernel1": 5, "kernel2": 3, "pool": 2, "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.0},
    {"model": "CNN1D", "lookback": 60, "filters1": 192, "filters2": 192, "kernel1": 5, "kernel2": 5, "pool": 2, "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
    {"model": "CNN1D", "lookback": 100,"filters1": 128, "filters2": 256, "kernel1": 7, "kernel2": 3, "pool": 2, "dropout": 0.4, "lr": 5e-4, "activation": "relu", "l2": 5e-4, "label_smoothing": 0.05},
]

resultados_val = {}
best_val_f1_mlp = -1.0
best_val_f1_cnn = -1.0

def make_callbacks(run_best_path: str):
    es = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ckpt = callbacks.ModelCheckpoint(run_best_path, monitor="val_accuracy", mode="max",
                                     save_best_only=True, save_weights_only=False)
    return [es, rlr, ckpt]

# Loop MLP
for i, hp in enumerate(mlp_space, start=1):
    run_name = f"MLP_hp{i}"
    with mlflow.start_run(run_name=run_name) as run:
        # Log params hp + de entorno
        mlflow.log_params({
            **hp,
            "n_features": X_tr_mlp.shape[1],
            "epochs": epochs, "batch_size": batch_size, "patience": patience
        })
        pesos = class_weights_balanced(y_tr_mlp)
        model_mlp = build_mlp(input_dim=X_tr_mlp.shape[1], num_classes=n_clases, params=hp)
        cbs = make_callbacks(os.path.join(outdir, f"best_mlp_hp{i}.keras"))

        hist = model_mlp.fit(
            X_tr_mlp, y_tr_mlp,
            validation_data=(X_va_mlp, y_va_mlp),
            epochs=epochs, batch_size=batch_size,
            class_weight=pesos,
            callbacks=cbs, verbose=2
        )

        # Eval
        proba_val = model_mlp.predict(X_va_mlp, verbose=0)
        proba_te  = model_mlp.predict(X_te_mlp, verbose=0)
        m_val = evaluate_split(y_va_mlp, proba_val, f"val_mlp_hp{i}")
        m_te  = evaluate_split(y_te_mlp, proba_te,  f"test_mlp_hp{i}")

        resultados_val[f"MLP_hp{i}"] = m_val

        # Si es el mejor MLP hasta ahora, exporta como best_mlp.keras (para main)
        if m_val["f1_macro"] > best_val_f1_mlp:
            best_val_f1_mlp = m_val["f1_macro"]
            # el checkpoint ya guardó el mejor por val_accuracy; lo reutilizamos como "oficial"
            src = os.path.join(outdir, f"best_mlp_hp{i}.keras")
            dst = os.path.join(outdir, "best_mlp.keras")
            try:
                import shutil
                shutil.copyfile(src, dst)
            except Exception as _e:
                pass

# Loop CNN (regenerando secuencias por HP set)
for j, hp in enumerate(cnn_space, start=1):
    run_name = f"CNN1D_hp{j}"
    lookback_hp = int(hp["lookback"])

    # Re-generar secuencias para este lookback
    X_tr_seq, y_tr_seq_raw = make_sequences(train_scaled, feat_cols, lookback_hp)
    X_va_seq, y_va_seq_raw = make_sequences(val_scaled,   feat_cols, lookback_hp)
    X_te_seq, y_te_seq_raw = make_sequences(test_scaled,  feat_cols, lookback_hp)

    # Remapeo coherente
    y_tr_seq = np.vectorize(map_mlp.get)(y_tr_seq_raw).astype(int)
    y_va_seq = np.vectorize(map_mlp.get)(y_va_seq_raw).astype(int)
    y_te_seq = np.vectorize(map_mlp.get)(y_te_seq_raw).astype(int)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            **hp,
            "n_features": X_tr_seq.shape[2],
            "epochs": epochs, "batch_size": batch_size, "patience": patience
        })
        pesos = class_weights_balanced(y_tr_seq)
        model_cnn = build_cnn(input_len=X_tr_seq.shape[1], num_features=X_tr_seq.shape[2],
                              num_classes=n_clases, params=hp)
        cbs = make_callbacks(os.path.join(outdir, f"best_cnn_hp{j}.keras"))

        hist = model_cnn.fit(
            X_tr_seq, y_tr_seq,
            validation_data=(X_va_seq, y_va_seq),
            epochs=epochs, batch_size=batch_size,
            class_weight=pesos,
            callbacks=cbs, verbose=2
        )

        # Eval
        proba_val = model_cnn.predict(X_va_seq, verbose=0)
        proba_te  = model_cnn.predict(X_te_seq, verbose=0)
        m_val = evaluate_split(y_va_seq, proba_val, f"val_cnn_hp{j}")
        m_te  = evaluate_split(y_te_seq,  proba_te,  f"test_cnn_hp{j}")

        resultados_val[f"CNN_hp{j}"] = m_val

        # Si es el mejor CNN hasta ahora, exporta como best_cnn.keras (para main)
        if m_val["f1_macro"] > best_val_f1_cnn:
            best_val_f1_cnn = m_val["f1_macro"]
            src = os.path.join(outdir, f"best_cnn_hp{j}.keras")
            dst = os.path.join(outdir, "best_cnn.keras")
            try:
                import shutil
                shutil.copyfile(src, dst)
            except Exception as _e:
                pass



# Selección del mejor por F1 en VALID 
mejor = max(resultados_val.items(), key=lambda kv: kv[1]["f1_macro"])[0]
print(f"Mejor modelo (VALID F1): {mejor}")
with mlflow.start_run(run_name=f"BEST_{mejor}_TEST") as run:
    mlflow.log_param("best_model", mejor)
