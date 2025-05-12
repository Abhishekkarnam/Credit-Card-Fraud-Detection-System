# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import traceback # Import traceback
import sqlite3 # Import SQLite
import json # Import JSON for storing dicts
from datetime import datetime # Import datetime for timestamps

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# PennyLane imports
import pennylane as qml
from pennylane import numpy as pnp # Use PennyLane's numpy for parameters

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Model Comparison for Fraud Detection",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables / Constants ---
TARGET = 'is_fraud'
# Define potential features - these will be filtered based on actual columns after loading/FE
POTENTIAL_NUMERICAL_FEATURES = [
    'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
    'trans_hour', 'trans_dayofweek', 'trans_month', 'age', 'distance_km'
]
POTENTIAL_CATEGORICAL_FEATURES = ['category', 'state']
# FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES list defined dynamically after data loading checks
DB_NAME = "predictions_log.db"

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and prediction_log table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            input_features_json TEXT NOT NULL,
            fraud_probability REAL NOT NULL,
            threshold REAL NOT NULL,
            prediction_value INTEGER NOT NULL,
            prediction_label TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Call init_db once when the script starts or when relevant
# For a Streamlit app, it's generally safe to call it here if the DB file is local.
# If deploying, ensure the DB path is writable.
try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize database: {e}")


# --- Load Data Function ---
@st.cache_data
def load_data(filepath):
    """Loads and performs initial cleaning and feature engineering."""
    try:
        df = pd.read_csv("C:\SEM4\Agile\datalab_export_2025-04-16 15_55_17.csv", encoding='utf-8-sig')
        df.columns = df.columns.str.replace('"', '', regex=False).str.replace(' ', '_', regex=False).str.lower()

        try:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
            if 'trans_date_trans_time' in df.columns:
                 df['trans_hour'] = df['trans_date_trans_time'].dt.hour
                 df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
                 df['trans_month'] = df['trans_date_trans_time'].dt.month
            else:
                 st.warning("'trans_date_trans_time' not found, cannot create time features.")

            if 'trans_date_trans_time' in df.columns and 'dob' in df.columns:
                valid_dates = df['trans_date_trans_time'].notna() & df['dob'].notna()
                df['age'] = np.nan
                df.loc[valid_dates, 'age'] = ((df.loc[valid_dates, 'trans_date_trans_time'] - df.loc[valid_dates, 'dob']).dt.days / 365.25)
                df['age'] = df['age'].apply(lambda x: x if pd.notna(x) and x > 0 else np.nan)
                median_age = df['age'].median()
                if pd.notna(median_age):
                    df['age'].fillna(median_age, inplace=True)
                else:
                    df['age'].fillna(0, inplace=True) # Fallback
                    st.warning("Could not calculate median age, filled NaNs with 0.")
            else:
                 st.warning("'trans_date_trans_time' or 'dob' not found, cannot create age feature.")
                 if 'age' not in df.columns: df['age'] = 0.0

        except KeyError as e:
            st.warning(f"Date column missing for FE: {e}. Skipping specific date features.")
        except Exception as e:
            st.error(f"Error during date FE: {e}")


        coord_cols = ['lat', 'long', 'merch_lat', 'merch_long']
        if all(col in df.columns for col in coord_cols):
            try:
                lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
                lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
                epsilon = 1e-9
                lat1 = np.clip(lat1, -np.pi/2 + epsilon, np.pi/2 - epsilon)
                lat2 = np.clip(lat2, -np.pi/2 + epsilon, np.pi/2 - epsilon)
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
                a = np.clip(a, 0, 1)
                c = 2 * np.arcsin(np.sqrt(a))
                r = 6371
                df['distance_km'] = c * r
                df['distance_km'].replace([np.inf, -np.inf], np.nan, inplace=True)
                median_dist = df['distance_km'].median()
                if pd.notna(median_dist):
                    df['distance_km'].fillna(median_dist, inplace=True)
                else:
                    df['distance_km'].fillna(0, inplace=True)
                    st.warning("Could not calculate median distance, filled NaNs with 0.")
            except Exception as e:
                 st.error(f"Error calculating distance: {e}")
                 if 'distance_km' not in df.columns: df['distance_km'] = 0.0
        else:
             st.warning(f"One or more coordinate columns {coord_cols} missing. Skipping distance_km feature.")
             if 'distance_km' not in df.columns: df['distance_km'] = 0.0


        cols_to_drop_post_fe = ['trans_date_trans_time', 'dob', 'lat', 'long', 'merch_lat', 'merch_long',
                                'index', 'trans_num', 'merchant', 'city', 'job']
        df.drop(columns=[col for col in cols_to_drop_post_fe if col in df.columns], inplace=True, errors='ignore')

        if TARGET in df.columns:
            if df[TARGET].isnull().any():
                 st.warning(f"NaNs found in target column '{TARGET}'. Attempting to fill with mode (0).")
                 df[TARGET].fillna(0, inplace=True)
            try:
                df[TARGET] = df[TARGET].astype(int)
            except ValueError as e:
                 st.error(f"Could not convert target column '{TARGET}' to int: {e}")
                 df[TARGET] = 0
                 st.warning(f"Target column '{TARGET}' conversion failed, set to 0.")

        potential_features_check = POTENTIAL_CATEGORICAL_FEATURES + POTENTIAL_NUMERICAL_FEATURES
        for col in potential_features_check:
             if col in df.columns and df[col].isnull().any():
                 if pd.api.types.is_numeric_dtype(df[col]):
                     med = df[col].median()
                     df[col].fillna(med if pd.notna(med) else 0, inplace=True)
                 else:
                     if not df[col].mode().empty:
                         df[col].fillna(df[col].mode()[0], inplace=True)
                     else:
                         df[col].fillna("Unknown", inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.error(traceback.format_exc())
        return None

data = load_data("datalab_export_2025-04-16 15_55_17.csv")

if data is not None:
    all_cols = data.columns.tolist()
    NUMERICAL_FEATURES = [f for f in POTENTIAL_NUMERICAL_FEATURES if f in all_cols and pd.api.types.is_numeric_dtype(data[f])]
    CATEGORICAL_FEATURES = [f for f in POTENTIAL_CATEGORICAL_FEATURES if f in all_cols and not pd.api.types.is_numeric_dtype(data[f])]
    FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    if TARGET not in all_cols:
        st.error(f"FATAL: Target column '{TARGET}' not found in the final DataFrame!")
        data = None
    elif not FEATURES:
        st.error("FATAL: No valid features found in the final DataFrame!")
        data = None
else:
     st.error("Data loading failed, cannot proceed.")
     st.stop()


def create_preprocessor(numerical_features, categorical_features):
    transformers = []
    if numerical_features:
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('num', numerical_transformer, numerical_features))
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    if not transformers:
         raise ValueError("No numerical or categorical features specified for preprocessing!")
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def plot_confusion_matrix_func(cm, classes):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    return fig

def plot_precision_recall_curve_func(y_true, y_scores, model_name="Model"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_val = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, marker='.', label=f'{model_name} (AUC = {pr_auc_val:.2f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'); ax.legend(); ax.grid(True)
    return fig

class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super(FraudNet, self).__init__()
        if input_dim <= 0:
            raise ValueError(f"Invalid input_dim for FraudNet: {input_dim}")
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
    def forward(self, inputs):
        if inputs.shape[0] > 1:
            x = self.relu(self.batchnorm1(self.layer_1(inputs)))
            x = self.dropout(x)
            x = self.relu(self.batchnorm2(self.layer_2(x)))
            x = self.dropout(x)
        else:
            x = self.relu(self.layer_1(inputs))
            x = self.dropout(x)
            x = self.relu(self.layer_2(x))
            x = self.dropout(x)
        x = self.layer_out(x)
        return x

def train_pytorch_model(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    epoch_losses = []
    status_text_train = st.empty()
    progress_bar_train = st.progress(0)
    start_time = time.time()
    total_batches = len(train_loader)
    if total_batches == 0: raise ValueError("Train loader is empty.")
    for epoch in range(epochs):
        current_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                 st.warning(f"NaN loss detected at epoch {epoch+1}, batch {i}. Stopping training.")
                 raise ValueError("NaN loss detected during training.")
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            progress = ((epoch * total_batches + i + 1) / (epochs * total_batches))
            progress_bar_train.progress(min(int(progress * 100), 100))
        avg_loss = current_loss / total_batches
        epoch_losses.append(avg_loss)
        status_text_train.text(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    end_time = time.time()
    training_time = end_time - start_time
    status_text_train.text(f"PyTorch Training Complete. Time: {training_time:.2f}s")
    progress_bar_train.empty()
    return model, epoch_losses, training_time

def evaluate_pytorch_model(model, test_loader, device):
    model.eval()
    y_pred_list, y_pred_proba_list, y_test_list = [], [], []
    if len(test_loader) == 0: raise ValueError("Test loader is empty.")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probas > 0.5).astype(int)
            y_pred_list.extend(preds)
            y_pred_proba_list.extend(probas)
            y_test_list.extend(labels.cpu().numpy())
    return np.array(y_test_list), np.array(y_pred_list), np.array(y_pred_proba_list)

try:
    q_dev = qml.device("lightning.qubit", wires=4)
except Exception:
    st.sidebar.warning("PennyLane Lightning backend failed/not found. Using default.qubit.")
    q_dev = qml.device("default.qubit", wires=4)
n_qubits = len(q_dev.wires)

@qml.qnode(q_dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class HybridModel(nn.Module):
    def __init__(self, n_features, n_qlayers=1):
        super().__init__()
        if n_features <= 0:
             raise ValueError(f"Invalid n_features for HybridModel: {n_features}")
        self.classical_layer = nn.Linear(n_features, n_qubits)
        self.relu = nn.ReLU()
        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.output_layer = nn.Linear(1, 1)
    def forward(self, x):
        x = self.relu(self.classical_layer(x))
        x = torch.tanh(x) * torch.pi
        x = self.qlayer(x)
        x = x.unsqueeze(-1) if x.ndim == 1 else x
        x = self.output_layer(x)
        return x

def train_qml_model(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    epoch_losses = []
    status_text_qml = st.empty()
    progress_bar_qml = st.progress(0)
    start_time = time.time()
    total_batches = len(train_loader)
    if total_batches == 0: raise ValueError("QML Train loader is empty.")
    status_text_qml.warning("QML Training can be slow...")
    for epoch in range(epochs):
        current_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                 st.warning(f"QML NaN loss detected at epoch {epoch+1}, batch {i}. Stopping.")
                 raise ValueError("NaN loss detected during QML training.")
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            progress = ((epoch * total_batches + i + 1) / (epochs * total_batches))
            progress_bar_qml.progress(min(int(progress * 100), 100))
        avg_loss = current_loss / total_batches
        epoch_losses.append(avg_loss)
        status_text_qml.text(f"QML Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    end_time = time.time()
    training_time = end_time - start_time
    status_text_qml.success(f"QML Training Complete. Time: {training_time:.2f}s")
    progress_bar_qml.empty()
    return model, epoch_losses, training_time

st.title("⚖️ ML/DL/QML Model Comparison for Fraud Detection")

if 'results' not in st.session_state: st.session_state['results'] = {}
if 'models' not in st.session_state: st.session_state['models'] = {}
if 'preprocessor' not in st.session_state: st.session_state['preprocessor'] = None
if 'input_feature_count' not in st.session_state: st.session_state['input_feature_count'] = None

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose a section:", ["Data Overview", "Model Training & Comparison", "Make Prediction", "View Prediction Log"])

if app_mode == "Data Overview":
    st.header("📊 Data Overview")
    if data is not None:
        st.write("Sample of Processed Data:")
        st.dataframe(data.head(10))
        st.subheader("Dataset Info")
        st.write(f"Shape: {data.shape}")
        if 'FEATURES' in globals() and FEATURES:
            st.write(f"Features for Modeling ({len(FEATURES)}): {FEATURES}")
        else:
            st.warning("Feature list not defined or empty.")
        st.write(f"Target: {TARGET}")

        st.subheader("Numerical Stats")
        if NUMERICAL_FEATURES:
             st.dataframe(data[NUMERICAL_FEATURES].describe())
        else: st.write("No numerical features identified.")

        st.subheader(f"Fraud Distribution ('{TARGET}')")
        if TARGET in data.columns:
            fraud_counts = data[TARGET].value_counts()
            st.write(fraud_counts)
            if not fraud_counts.empty and data.shape[0] > 0:
                fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
                sns.countplot(x=TARGET, data=data, ax=ax_dist, palette="viridis", hue=TARGET, legend=False)
                ax_dist.set_title(f"Distribution of Fraud Target (0: Not Fraud, 1: Fraud)")
                ax_dist.set_xlabel("Fraud Status"); ax_dist.set_ylabel("Count")
                ax_dist.set_xticks([0, 1]); ax_dist.set_xticklabels(['Not Fraud (0)', 'Fraud (1)'])
                st.pyplot(fig_dist)
                if data.shape[0] > 0:
                    fraud_rate = fraud_counts.get(1, 0) / data.shape[0] * 100
                    st.caption(f"Fraud rate: {fraud_rate:.2f}%")
                else:
                    st.caption("Fraud rate: N/A (empty dataset)")
            else:
                st.warning("Target column data insufficient for plot.")
        else:
             st.warning("Target column not found.")

        st.subheader("Correlation Heatmap (Numerical)")
        if NUMERICAL_FEATURES and TARGET in data.columns:
            numeric_cols_for_corr = data[NUMERICAL_FEATURES + [TARGET]].select_dtypes(include=np.number)
            if not numeric_cols_for_corr.empty:
                numeric_cols_for_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
                if numeric_cols_for_corr.isnull().any().any():
                    st.warning("NaNs/Infs found in numeric data for correlation. Imputing with median.")
                    for col in numeric_cols_for_corr.columns:
                        if numeric_cols_for_corr[col].isnull().any():
                             med = numeric_cols_for_corr[col].median()
                             numeric_cols_for_corr[col].fillna(med if pd.notna(med) else 0, inplace=True)
                corr = numeric_cols_for_corr.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(max(10, len(NUMERICAL_FEATURES)*0.8), max(8, len(NUMERICAL_FEATURES)*0.6)))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
                plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
                ax_corr.set_title("Correlation Matrix")
                st.pyplot(fig_corr)
            else:
                st.write("No numeric columns suitable for correlation.")
        else:
            st.write("Numerical features or target column needed for correlation.")
    else:
        st.error("Data object is None, cannot display overview.")


elif app_mode == "Model Training & Comparison":
    st.header("🤖 Model Training & Comparison")
    if data is None:
         st.error("Cannot train models, data loading failed.")
         st.stop()

    model_type = st.selectbox("Select Model:", ["Random Forest", "PyTorch NN", "Quantum Circuit (Hybrid)"])

    st.subheader("Shared Parameters")
    test_size = st.slider("Test Size (%)", 10, 50, 30, 5, key='shared_test_size') / 100.0
    random_state = st.number_input("Random State", value=42, step=1, key='shared_random_state')

    st.subheader(f"{model_type} Hyperparameters")
    if model_type == "Random Forest":
        n_estimators = st.slider("N Estimators", 50, 300, 100, 10, key='rf_n_est')
        max_depth_rf = st.slider("Max Depth (0=None)", 0, 30, 10, 1, key='rf_max_depth')
        min_samples_leaf_rf = st.slider("Min Samples Leaf", 1, 50, 10, 1, key='rf_min_leaf')
        use_class_weight_rf = st.checkbox("Use Class Weight", True, key='rf_class_weight')
    elif model_type == "PyTorch NN":
        lr_pytorch = st.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, 1e-4, format="%f", key='pt_lr')
        epochs_pytorch = st.number_input("Epochs", 1, 50, 10, 1, key='pt_epochs')
        batch_size_pytorch = st.select_slider("Batch Size", [32, 64, 128, 256], 64, key='pt_batch_size')
    elif model_type == "Quantum Circuit (Hybrid)":
        st.warning("QML training is experimental & slow.")
        lr_qml = st.number_input("Learning Rate", 1e-4, 1e-1, 5e-3, 1e-4, format="%f", key='qml_lr')
        epochs_qml = st.number_input("Epochs", 1, 20, 5, 1, key='qml_epochs')
        batch_size_qml = st.select_slider("Batch Size", [16, 32, 64], 32, key='qml_batch_size')
        n_qlayers_qml = st.number_input("Quantum Layers", 1, 4, 1, 1, key='qml_layers')

    if st.button(f"Train {model_type} Model", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            X = data[FEATURES].copy()
            y = data[TARGET].copy()
            if X.isnull().any().any() or y.isnull().any():
                 st.warning("NaNs detected before split. Performing final imputation...")
                 for col in X.columns:
                     if X[col].isnull().any():
                         if pd.api.types.is_numeric_dtype(X[col]):
                             med = X[col].median()
                             X[col].fillna(med if pd.notna(med) else 0, inplace=True)
                         else:
                             mode_val = X[col].mode()
                             X[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)
                 if y.isnull().any(): y.fillna(0, inplace=True)
                 if X.isnull().any().any():
                      st.error("NaN imputation failed for features. Cannot proceed.")
                      st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            progress_bar.progress(10)

            if st.session_state.get('preprocessor') is None:
                status_text.text("Fitting preprocessor...")
                if not NUMERICAL_FEATURES and not CATEGORICAL_FEATURES:
                     raise ValueError("Cannot create preprocessor: Both numerical and categorical feature lists are empty.")
                preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
                preprocessor.fit(X_train)
                st.session_state['preprocessor'] = preprocessor
                try:
                    sample_transformed = preprocessor.transform(X_train.head(1))
                    st.session_state['input_feature_count'] = sample_transformed.shape[1]
                except Exception as e_shape:
                    st.error(f"Could not determine shape after preprocessing: {e_shape}"); st.stop()
            else:
                preprocessor = st.session_state['preprocessor']
                if st.session_state.get('input_feature_count') is None:
                    try:
                         sample_transformed = preprocessor.transform(X_train.head(1))
                         st.session_state['input_feature_count'] = sample_transformed.shape[1]
                    except Exception as e_shape:
                         st.error(f"Could not determine shape with existing preprocessor: {e_shape}"); st.stop()

            input_dim = st.session_state.get('input_feature_count')
            if input_dim is None or input_dim <= 0:
                raise ValueError(f"Invalid input dimension determined after preprocessing: {input_dim}")

            status_text.text("Preprocessing training/test data...")
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            progress_bar.progress(20)
            y_train_np = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
            y_test_np = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)

            model, training_time = None, 0
            y_pred, y_pred_proba = None, None

            if model_type == "Random Forest":
                status_text.text(f"Training Random Forest...")
                max_depth_val_rf = None if max_depth_rf == 0 else max_depth_rf
                class_weight_rf = 'balanced' if use_class_weight_rf else None
                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_val_rf, min_samples_leaf=min_samples_leaf_rf, class_weight=class_weight_rf, random_state=random_state, n_jobs=-1)
                start_time = time.time(); rf_model.fit(X_train_processed, y_train_np); training_time = time.time() - start_time
                status_text.success(f"RF trained ({training_time:.2f}s)"); progress_bar.progress(80)
                y_pred = rf_model.predict(X_test_processed); y_pred_proba = rf_model.predict_proba(X_test_processed)[:, 1]
                model = rf_model

            elif model_type == "PyTorch NN":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                status_text.text(f"Training PyTorch NN on {device}..."); progress_bar.progress(30)
                pt_model = FraudNet(input_dim=input_dim).to(device)
                criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(pt_model.parameters(), lr=lr_pytorch)
                train_dataset = TensorDataset(torch.tensor(X_train_processed).float(), torch.tensor(y_train_np).float())
                test_dataset = TensorDataset(torch.tensor(X_test_processed).float(), torch.tensor(y_test_np).float())
                train_loader = DataLoader(train_dataset, batch_size=batch_size_pytorch, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size_pytorch, shuffle=False)
                pt_model, _, training_time = train_pytorch_model(pt_model, train_loader, optimizer, criterion, device, epochs=epochs_pytorch)
                progress_bar.progress(80)
                _, y_pred, y_pred_proba = evaluate_pytorch_model(pt_model, test_loader, device)
                model = pt_model

            elif model_type == "Quantum Circuit (Hybrid)":
                if input_dim < n_qubits: raise ValueError(f"Processed features ({input_dim}) < n_qubits ({n_qubits}). Adjust classical layer or n_qubits.")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                status_text.text(f"Training QML on {device}..."); progress_bar.progress(30)
                qml_model = HybridModel(n_features=input_dim, n_qlayers=n_qlayers_qml).to(device)
                criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(qml_model.parameters(), lr=lr_qml)
                train_dataset = TensorDataset(torch.tensor(X_train_processed).float(), torch.tensor(y_train_np).float())
                test_dataset = TensorDataset(torch.tensor(X_test_processed).float(), torch.tensor(y_test_np).float())
                train_loader = DataLoader(train_dataset, batch_size=batch_size_qml, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size_qml, shuffle=False)
                qml_model, _, training_time = train_qml_model(qml_model, train_loader, optimizer, criterion, device, epochs=epochs_qml)
                progress_bar.progress(80)
                _, y_pred, y_pred_proba = evaluate_pytorch_model(qml_model, test_loader, device)
                model = qml_model

            if y_pred is None or y_pred_proba is None:
                 raise ValueError("Model evaluation failed to produce predictions.")

            status_text.text("Calculating metrics...")
            accuracy = accuracy_score(y_test_np, y_pred)
            report_dict = classification_report(y_test_np, y_pred, target_names=['Not Fraud (0)', 'Fraud (1)'], output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test_np, y_pred)
            roc_auc_val, pr_auc_val = np.nan, np.nan
            if len(np.unique(y_test_np)) > 1:
                roc_auc_val = roc_auc_score(y_test_np, y_pred_proba)
                precision, recall, _ = precision_recall_curve(y_test_np, y_pred_proba)
                pr_auc_val = auc(recall, precision)
            else:
                st.warning("Only one class present in y_test. ROC AUC and PR AUC not calculated.")
            progress_bar.progress(100)
            status_text.success(f"{model_type} evaluation complete.")

            st.subheader(f"Results for {model_type}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("ROC AUC", f"{roc_auc_val:.4f}" if pd.notna(roc_auc_val) else "N/A")
            col3.metric("Training Time (s)", f"{training_time:.2f}")
            st.text("Classification Report:")
            st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.4f}"))
            col_cm, col_pr = st.columns(2)
            with col_cm:
                st.text("Confusion Matrix:")
                fig_cm = plot_confusion_matrix_func(cm, classes=['Not Fraud', 'Fraud'])
                st.pyplot(fig_cm)
            with col_pr:
                 st.text("Precision-Recall Curve:")
                 if pd.notna(pr_auc_val):
                     fig_pr = plot_precision_recall_curve_func(y_test_np, y_pred_proba, model_name=model_type)
                     st.pyplot(fig_pr)
                 else:
                     st.write("PR curve N/A (only one class in test set).")

            results_summary = {
                'Accuracy': accuracy, 'ROC_AUC': roc_auc_val, 'PR_AUC': pr_auc_val,
                'Precision (Fraud)': report_dict.get('Fraud (1)', {}).get('precision', 0.0),
                'Recall (Fraud)': report_dict.get('Fraud (1)', {}).get('recall', 0.0),
                'F1-Score (Fraud)': report_dict.get('Fraud (1)', {}).get('f1-score', 0.0),
                'Training Time (s)': training_time
            }
            st.session_state['results'][model_type] = results_summary
            st.session_state['models'][model_type] = model
            st.success(f"{model_type} model & results stored.")

        except Exception as e:
            st.error(f"An error occurred during {model_type} training/evaluation: {e}")
            st.error(traceback.format_exc())
            status_text.error("Training failed.")
            progress_bar.empty()

    st.subheader("Model Comparison Summary")
    if st.session_state['results']:
        comparison_df = pd.DataFrame(st.session_state['results']).T.fillna("N/A")
        numeric_cols = ['Accuracy', 'ROC_AUC', 'PR_AUC', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'Training Time (s)']
        perf_cols = [col for col in numeric_cols if col != 'Training Time (s)' and col in comparison_df.columns]
        time_col = ['Training Time (s)'] if 'Training Time (s)' in comparison_df.columns else []
        styled_df = comparison_df.style.format("{:.4f}", subset=pd.IndexSlice[:, perf_cols + time_col], na_rep="N/A")
        if perf_cols:
            styled_df = styled_df.highlight_max(axis=0, color='lightgreen', subset=perf_cols)
        if time_col:
            styled_df = styled_df.highlight_min(axis=0, color='#d65f5f', subset=time_col)
        st.dataframe(styled_df)
    else:
        st.info("Train models to see comparison.")

elif app_mode == "Make Prediction":
    st.header("🔮 Make Prediction")
    if not st.session_state.get('models') or st.session_state.get('preprocessor') is None:
        st.warning("Train at least one model first in the 'Model Training & Comparison' section.")
        st.stop()

    available_models = list(st.session_state['models'].keys())
    if not available_models:
        st.warning("No trained models available for prediction.")
        st.stop()

    selected_model_name = st.selectbox("Select Model:", available_models)
    model_to_predict = st.session_state['models'].get(selected_model_name)
    preprocessor = st.session_state['preprocessor']

    if model_to_predict is None or preprocessor is None:
        st.error("Selected model or preprocessor is missing from session state. Please retrain.")
        st.stop()

    st.write(f"Using **{selected_model_name}** model.")
    st.write("Enter transaction details:")
    input_data = {}
    cols = st.columns(3)
    current_col_idx = 0
    data_for_inputs = data # Use original loaded data for input ranges/options

    for feature in FEATURES:
        target_col = cols[current_col_idx % 3]
        with target_col:
            feature_label = feature.replace('_', ' ').title()
            if feature in CATEGORICAL_FEATURES:
                options = ["Unknown"]
                if feature in data_for_inputs.columns:
                    unique_vals = data_for_inputs[feature].dropna().unique()
                    if len(unique_vals) > 0: options = sorted(list(unique_vals))
                input_data[feature] = st.selectbox(f"{feature_label}:", options=options, key=f"input_{feature}")
            elif feature in NUMERICAL_FEATURES:
                min_val, max_val, median_val, step_val = 0.0, 1000.0, 50.0, 1.0
                if feature in data_for_inputs.columns:
                    try:
                        f_data = data_for_inputs[feature].dropna()
                        if not f_data.empty:
                            min_val = float(f_data.min()); max_val = float(f_data.max())
                            median_val = float(f_data.median())
                            range_val = max_val - min_val
                            if range_val > 0: step_val = max(1e-4, range_val / 100)
                            else: step_val = 1.0
                    except Exception as e:
                        st.warning(f"Stats error for {feature}: {e}. Using defaults.")
                input_data[feature] = st.number_input(f"{feature_label}:", min_value=min_val, max_value=max_val, value=median_val, step=step_val, key=f"input_{feature}", format="%f")
        current_col_idx += 1

    pred_threshold = st.slider("Prediction Threshold", 0.05, 0.95, 0.5, 0.05, key='pred_thresh')

    if st.button(f"Predict Fraud Risk with {selected_model_name}", type="primary"):
        input_df = pd.DataFrame([input_data])[FEATURES]

        try:
            if input_df.isnull().any().any():
                 st.warning("NaNs detected in user input. Imputing...")
                 for col in input_df.columns:
                     if input_df[col].isnull().any():
                         if pd.api.types.is_numeric_dtype(input_df[col]):
                             med = data[col].median()
                             input_df[col].fillna(med if pd.notna(med) else 0, inplace=True)
                         else:
                             mode_val = data[col].mode()
                             input_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)

            st.write("Input DataFrame for Preprocessing:")
            st.dataframe(input_df)

            input_processed = preprocessor.transform(input_df)
            fraud_probability = 0.0

            if selected_model_name == "Random Forest":
                fraud_probability = model_to_predict.predict_proba(input_processed)[0, 1]
            elif selected_model_name == "PyTorch NN" or selected_model_name == "Quantum Circuit (Hybrid)":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_to_predict.to(device).eval()
                with torch.no_grad():
                    input_tensor = torch.tensor(input_processed).float().to(device)
                    output = model_to_predict(input_tensor)
                    prob = torch.sigmoid(output).cpu().numpy().flatten()
                    fraud_probability = prob[0]

            prediction_value = 1 if fraud_probability >= pred_threshold else 0
            prediction_label_text = "🚨 FRAUD Risk Detected" if prediction_value == 1 else "✅ Legitimate Transaction Likely"


            st.subheader("Prediction Result")
            st.metric(f"Fraud Probability ({selected_model_name})", f"{fraud_probability:.4f}")
            if prediction_value == 1: st.error(f"{prediction_label_text} (Prob >= {pred_threshold:.2f})!")
            else: st.success(f"{prediction_label_text} (Prob < {pred_threshold:.2f}).")

            # Save prediction to database
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Convert input_data (which might contain numpy types) to JSON serializable types
                serializable_input_data = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in input_data.items()}
                input_features_str = json.dumps(serializable_input_data)

                cursor.execute("""
                    INSERT INTO prediction_log (
                        timestamp, model_name, input_features_json,
                        fraud_probability, threshold, prediction_value, prediction_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp_now, selected_model_name, input_features_str,
                    float(fraud_probability), float(pred_threshold), int(prediction_value),
                    prediction_label_text.split('(')[0].strip() # Get the core label
                ))
                conn.commit()
                conn.close()
                st.success("Prediction details saved to the database log.")
            except Exception as db_e:
                st.error(f"Error saving prediction to database: {db_e}")
                st.error(traceback.format_exc())


        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.error(traceback.format_exc())

elif app_mode == "View Prediction Log":
    st.header("📜 View Prediction Log")
    try:
        conn = sqlite3.connect(DB_NAME)
        # Query to select specific columns and rename prediction_label for clarity
        query = "SELECT id, timestamp, model_name, fraud_probability, threshold, prediction_value, prediction_label AS status, input_features_json FROM prediction_log ORDER BY timestamp DESC"
        log_df = pd.read_sql_query(query, conn)
        conn.close()

        if not log_df.empty:
            st.write(f"Displaying last {len(log_df)} predictions:")

            # Option to show/hide JSON input features
            show_json = st.checkbox("Show detailed input features (JSON)", False)

            if not show_json:
                cols_to_show = [col for col in log_df.columns if col != 'input_features_json']
                st.dataframe(log_df[cols_to_show])
            else:
                st.dataframe(log_df)

            st.download_button(
                label="Download Prediction Log as CSV",
                data=log_df.to_csv(index=False).encode('utf-8'),
                file_name='prediction_log.csv',
                mime='text/csv',
            )
        else:
            st.info("No predictions logged yet.")
    except Exception as e:
        st.error(f"Error reading prediction log from database: {e}")
        st.error(traceback.format_exc())


# Footer
st.sidebar.markdown("---")
st.sidebar.info("Fraud Detection Model Comparison")
# --- END OF FILE app.py ---