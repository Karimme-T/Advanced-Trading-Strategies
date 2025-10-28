import os, json, numpy as np
import mlflow, mlflow.tensorflow
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from Feature_eng import make_sequences, train_scaled, val_scaled, test_scaled, feat_cols
import time
import signal


# PARÁMETROS 
experiment_name = "dl_trading_fast"
lookback = 100      
epochs = 50              
batch_size = 252
patience = 5            
max_time_per_model = 600 
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)


class TimeoutCallback(callbacks.Callback):
    """Callback que detiene el entrenamiento después de max_time segundos"""
    def __init__(self, max_time_seconds):
        super().__init__()
        self.max_time = max_time_seconds
        self.start_time = None
        
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"⏱️  Timeout configurado: {self.max_time}s ({self.max_time/60:.1f} min)")
        
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_time:
            print(f"\n⚠️  TIMEOUT alcanzado después de {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"    Deteniendo entrenamiento en época {epoch+1}")
            self.model.stop_training = True


def make_sparse_ce(label_smoothing: float = 0.0):
    try:
        return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing and label_smoothing > 0:
            print("[WARN] label_smoothing no soportado; usando loss sin smoothing.")
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
    plt.imshow(cm, interpolation="nearest", cmap='Blues')
    plt.title(f"Matriz de confusión — {split_name}")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center', color='white' if v > cm.max()/2 else 'black')
    plt.colorbar()
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
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(rep_path)
    return {"accuracy": acc, "f1_macro": f1, "auc_macro_ovr": auc}


# MODELOS

def build_mlp(input_dim: int, num_classes: int = 3, params: dict | None = None) -> tf.keras.Model:
    params = params or {}
    hidden = params.get("hidden", [512, 128])
    drop = float(params.get("dropout", 0.2))
    lr = float(params.get("lr", 5e-3))
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
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


def build_cnn(input_len: int, num_features: int, num_classes: int = 3, params: dict | None = None) -> tf.keras.Model:
    params = params or {}
    f1 = int(params.get("filters1", 512))
    f2 = int(params.get("filters2", 128))
    k1 = int(params.get("kernel1", 5))
    k2 = int(params.get("kernel2", 5))
    pool = int(params.get("pool", 2))
    drop = float(params.get("dropout", 0.4))
    lr = float(params.get("lr", 1e-4))
    l2w = float(params.get("l2", 1e-4))
    act = params.get("activation", "softmax")
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
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


# PREPARACIÓN DE DATOS
print("Preparando datos...")

# Para MLP (día a día)
X_tr_mlp, y_tr_mlp, map_mlp = xy_daywise(train_scaled, feat_cols)
X_va_mlp, y_va_mlp, _        = xy_daywise(val_scaled,   feat_cols)
X_te_mlp, y_te_mlp, _        = xy_daywise(test_scaled,  feat_cols)
n_clases = len(np.unique(np.r_[y_tr_mlp, y_va_mlp, y_te_mlp]))

print(f"  MLP - Train: {X_tr_mlp.shape}, Val: {X_va_mlp.shape}, Test: {X_te_mlp.shape}")

# Para CNN 
X_tr_seq, y_tr_seq_raw = make_sequences(train_scaled, feat_cols, lookback)
X_va_seq, y_va_seq_raw = make_sequences(val_scaled,   feat_cols, lookback)
X_te_seq, y_te_seq_raw = make_sequences(test_scaled,  feat_cols, lookback)

# Remapeo coherente para CNN 
y_tr_seq = np.vectorize(map_mlp.get)(y_tr_seq_raw).astype(int)
y_va_seq = np.vectorize(map_mlp.get)(y_va_seq_raw).astype(int)
y_te_seq = np.vectorize(map_mlp.get)(y_te_seq_raw).astype(int)

print(f"  CNN - Train: {X_tr_seq.shape}, Val: {X_va_seq.shape}, Test: {X_te_seq.shape}")
print(f"  Clases: {n_clases}")
print()


# MLFLOW 
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


# HYPERPARAMETER SPACE (

mlp_space = [
    {"model": "MLP", "hidden": [256,128], "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
    {"model": "MLP", "hidden": [512,256], "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
]

cnn_space = [
    {"model": "CNN1D", "lookback": 60, "filters1": 128, "filters2": 256, "kernel1": 5, "kernel2": 3, "pool": 2, "dropout": 0.3, "lr": 1e-3, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
    {"model": "CNN1D", "lookback": 100,"filters1": 128, "filters2": 256, "kernel1": 5, "kernel2": 3, "pool": 2, "dropout": 0.4, "lr": 5e-4, "activation": "relu", "l2": 1e-4, "label_smoothing": 0.05},
]

resultados_val = {}
best_val_f1_mlp = -1.0
best_val_f1_cnn = -1.0


def make_callbacks(run_best_path: str):
    """Callbacks con TIMEOUT y early stopping agresivo"""
    es = callbacks.EarlyStopping(
        monitor="val_accuracy", 
        mode="max", 
        patience=patience,  
        restore_best_weights=True,
        verbose=1
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    ckpt = callbacks.ModelCheckpoint(
        run_best_path, 
        monitor="val_accuracy", 
        mode="max",
        save_best_only=True, 
        save_weights_only=False,
        verbose=0
    )
    timeout = TimeoutCallback(max_time_seconds=max_time_per_model)
    
    return [es, rlr, ckpt, timeout]


# ENTRENAMIENTO MLP
print("="*80)
print("ENTRENANDO MODELOS MLP")
print("="*80)

for i, hp in enumerate(mlp_space, start=1):
    run_name = f"MLP_hp{i}"
    print(f"\n[{i}/{len(mlp_space)}] {run_name}")
    print(f"   Config: {hp}")
    
    start_time = time.time()
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Log params
            mlflow.log_params({
                **hp,
                "n_features": X_tr_mlp.shape[1],
                "epochs": epochs, 
                "batch_size": batch_size, 
                "patience": patience,
                "max_time_per_model": max_time_per_model
            })
            
            pesos = class_weights_balanced(y_tr_mlp)
            model_mlp = build_mlp(input_dim=X_tr_mlp.shape[1], num_classes=n_clases, params=hp)
            cbs = make_callbacks(os.path.join(outdir, f"best_mlp_hp{i}.keras"))

            print("  Entrenando...")
            hist = model_mlp.fit(
                X_tr_mlp, y_tr_mlp,
                validation_data=(X_va_mlp, y_va_mlp),
                epochs=epochs, 
                batch_size=batch_size,
                class_weight=pesos,
                callbacks=cbs, 
                verbose=0  
            )

            elapsed = time.time() - start_time
            print(f"   Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"   Épocas completadas: {len(hist.history['loss'])}")

            # Eval
            print("   Evaluando...")
            proba_val = model_mlp.predict(X_va_mlp, verbose=0)
            proba_te  = model_mlp.predict(X_te_mlp, verbose=0)
            m_val = evaluate_split(y_va_mlp, proba_val, f"val_mlp_hp{i}")
            m_te  = evaluate_split(y_te_mlp, proba_te,  f"test_mlp_hp{i}")

            print(f"   Val  - Acc: {m_val['accuracy']:.4f}, F1: {m_val['f1_macro']:.4f}")
            print(f"   Test - Acc: {m_te['accuracy']:.4f}, F1: {m_te['f1_macro']:.4f}")

            resultados_val[f"MLP_hp{i}"] = m_val

            # Si es el mejor MLP
            if m_val["f1_macro"] > best_val_f1_mlp:
                best_val_f1_mlp = m_val["f1_macro"]
                src = os.path.join(outdir, f"best_mlp_hp{i}.keras")
                dst = os.path.join(outdir, "best_mlp.keras")
                try:
                    import shutil
                    shutil.copyfile(src, dst)
                    print(f"  NUEVO MEJOR MLP (F1={best_val_f1_mlp:.4f})")
                except Exception as _e:
                    pass
                    
    except Exception as e:
        print(f"  ERROR en {run_name}: {e}")
        import traceback
        traceback.print_exc()



# ENTRENAMIENTO CNN

print("\n" + "="*80)
print("ENTRENANDO MODELOS CNN")
print("="*80)

for j, hp in enumerate(cnn_space, start=1):
    run_name = f"CNN1D_hp{j}"
    lookback_hp = int(hp["lookback"])
    
    print(f"\n🔸 [{j}/{len(cnn_space)}] {run_name} (lookback={lookback_hp})")
    print(f"   Config: {hp}")
    
    start_time = time.time()

    try:
        # Re-generar secuencias para este lookback
        print(f"  Regenerando secuencias con lookback={lookback_hp}...")
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
                "epochs": epochs, 
                "batch_size": batch_size, 
                "patience": patience,
                "max_time_per_model": max_time_per_model
            })
            
            pesos = class_weights_balanced(y_tr_seq)
            model_cnn = build_cnn(
                input_len=X_tr_seq.shape[1], 
                num_features=X_tr_seq.shape[2],
                num_classes=n_clases, 
                params=hp
            )
            cbs = make_callbacks(os.path.join(outdir, f"best_cnn_hp{j}.keras"))

            print("   🏋️  Entrenando...")
            hist = model_cnn.fit(
                X_tr_seq, y_tr_seq,
                validation_data=(X_va_seq, y_va_seq),
                epochs=epochs, 
                batch_size=batch_size,
                class_weight=pesos,
                callbacks=cbs, 
                verbose=0  
            )

            elapsed = time.time() - start_time
            print(f"   Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"   Épocas completadas: {len(hist.history['loss'])}")

            # Eval
            print("   Evaluando...")
            proba_val = model_cnn.predict(X_va_seq, verbose=0)
            proba_te  = model_cnn.predict(X_te_seq, verbose=0)
            m_val = evaluate_split(y_va_seq, proba_val, f"val_cnn_hp{j}")
            m_te  = evaluate_split(y_te_seq,  proba_te,  f"test_cnn_hp{j}")

            print(f"   Val  - Acc: {m_val['accuracy']:.4f}, F1: {m_val['f1_macro']:.4f}")
            print(f"   Test - Acc: {m_te['accuracy']:.4f}, F1: {m_te['f1_macro']:.4f}")

            resultados_val[f"CNN_hp{j}"] = m_val

            # Si es el mejor CNN
            if m_val["f1_macro"] > best_val_f1_cnn:
                best_val_f1_cnn = m_val["f1_macro"]
                src = os.path.join(outdir, f"best_cnn_hp{j}.keras")
                dst = os.path.join(outdir, "best_cnn.keras")
                try:
                    import shutil
                    shutil.copyfile(src, dst)
                    print(f"   NUEVO MEJOR CNN (F1={best_val_f1_cnn:.4f})")
                except Exception as _e:
                    pass
                    
    except Exception as e:
        print(f"   ERROR en {run_name}: {e}")
        import traceback
        traceback.print_exc()


# RESUMEN FINAL
print("\n" + "="*80)
print("RESUMEN DE RESULTADOS")
print("="*80)

for nombre, metricas in sorted(resultados_val.items()):
    print(f"{nombre:15s} - Val F1: {metricas['f1_macro']:.4f}, Acc: {metricas['accuracy']:.4f}")

# Selección del mejor por F1 en VALID 
if resultados_val:
    mejor = max(resultados_val.items(), key=lambda kv: kv[1]["f1_macro"])[0]
    print(f"\n🏆 MEJOR MODELO (VALID F1): {mejor}")
    print(f"   F1-score: {resultados_val[mejor]['f1_macro']:.4f}")
    print(f"   Accuracy: {resultados_val[mejor]['accuracy']:.4f}")
    
    with mlflow.start_run(run_name=f"BEST_{mejor}_SUMMARY") as run:
        mlflow.log_param("best_model", mejor)
        mlflow.log_metrics(resultados_val[mejor])
else:
    print("No se completó ningún modelo exitosamente")

print("\nEntrenamiento completado!")
print(f"Modelos guardados en: {outdir}/")
print(f"Experimento MLflow: {experiment_name}")


mejor = max(resultados_val.items(), key=lambda kv: kv[1]["f1_macro"])[0]
print(f"MEJOR MODELO: {mejor}")

# Buscar el run del mejor modelo
experiment = mlflow.get_experiment_by_name(experiment_name)
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_f1_macro DESC"],
    max_results=1
)

best_run_id = runs_df.iloc[0]['run_id']


model_name = "amzn_trading_model"
model_uri = f"runs:/{best_run_id}/model"

try:
    # Registrar nueva versión del modelo
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "model_type": mejor.split('_')[0],  # MLP o CNN
            "val_f1_macro": resultados_val[mejor]["f1_macro"],
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
    )
    
    print(f"Modelo registrado: {model_name} version {model_version.version}")
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True  # Archiva versiones anteriores
    )
    
    print(f" Modelo promovido a Production")
    
except Exception as e:
    print(f"Error registrando modelo: {e}")
    print("El modelo se guardó localmente en outputs/")