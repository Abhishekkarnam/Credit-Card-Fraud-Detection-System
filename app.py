# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time # To simulate processing
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
print("hi")
# --- Load Data ---
# Use caching to load data only once
@st.cache_data
def load_data(filepath):
    """Loads and performs initial cleaning and feature engineering."""
    try:
        # Load the CSV, handle potential BOM (like ï»¿)
        df = pd.read_csv(filepath, encoding='utf-8-sig') # Use utf-8-sig for BOM
        # Clean column names (remove quotes, spaces, lowercase)
        df.columns = df.columns.str.replace('"', '', regex=False).str.replace(' ', '_', regex=False).str.lower()
        st.write("Original Columns:", df.columns.tolist()) # Debug: Show original columns

        # --- Feature Engineering ---
        st.write("Performing Feature Engineering...")

        # 1. Datetime Features
        try:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

            # Extract features from transaction time
            df['trans_hour'] = df['trans_date_trans_time'].dt.hour
            df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
            df['trans_month'] = df['trans_date_trans_time'].dt.month

            # Calculate Age at time of transaction
            # Ensure both dates are valid before calculation
            valid_dates = df['trans_date_trans_time'].notna() & df['dob'].notna()
            df['age'] = np.nan # Initialize age column
            # Calculate age in years (approximate)
            df.loc[valid_dates, 'age'] = ((df.loc[valid_dates, 'trans_date_trans_time'] - df.loc[valid_dates, 'dob']).dt.days / 365.25)
            # Handle potential invalid ages (e.g., negative if dob is after trans_date)
            df['age'] = df['age'].apply(lambda x: x if x is not None and x > 0 else np.nan)
            # Fill missing age with the median age (a common strategy)
            median_age = df['age'].median()
            df['age'].fillna(median_age, inplace=True)
            st.write(f"Filled missing age values with median: {median_age:.1f}")

        except Exception as e:
            st.error(f"Error during date/time feature engineering: {e}")
            st.warning("Date-related features might be missing or incorrect.")

        # 2. Location Features (Example: Distance between customer and merchant)
        try:
            # Calculate distance (Haversine formula requires radians)
            lat1, lon1 = np.radians(df['lat']), np.radians(df['long'])
            lat2, lon2 = np.radians(df['merch_lat']), np.radians(df['merch_long'])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371 # Radius of Earth in kilometers
            df['distance_km'] = c * r
            # Handle potential NaN distances (if lat/long missing) - fill with median
            median_dist = df['distance_km'].median()
            df['distance_km'].fillna(median_dist, inplace=True)
            st.write(f"Calculated customer-merchant distance. Filled missing with median: {median_dist:.2f} km")
        except Exception as e:
            st.error(f"Error calculating distance: {e}")
            st.warning("Distance feature might be missing or incorrect.")


        st.write("Feature Engineering Complete. Final Columns:", df.columns.tolist())
        return df

    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# Load the specific fraud detection dataset
data = load_data("datalab_export_2025-04-16 15_55_17.csv")

# --- Global Variables / Constants (Specific to Fraud Detection) ---
if data is not None:
    # Define features and target based on cleaned/engineered column names
    TARGET = 'is_fraud'

    # Feature Selection (Adjust based on analysis and desired complexity)
    NUMERICAL_FEATURES = [
        'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
        'trans_hour', 'trans_dayofweek', 'trans_month', 'age', 'distance_km'
    ]
    # Be cautious with high-cardinality features. Start simple.
    # 'merchant', 'city', 'job' have many unique values.
    # For this example, we'll use 'category' and 'state'.
    CATEGORICAL_FEATURES = ['category', 'state']

    # Ensure all selected features actually exist after engineering
    all_cols = data.columns.tolist()
    NUMERICAL_FEATURES = [f for f in NUMERICAL_FEATURES if f in all_cols]
    CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in all_cols]

    FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    # Columns to drop before training (original cols + ID cols + derived cols parents)
    DROP_COLS = [
        'index', 'trans_date_trans_time', 'merchant', 'city', 'job', 'dob',
        'trans_num', 'lat', 'long', 'merch_lat', 'merch_long' # Drop original lat/long if using distance
        # Keep 'category', 'state', 'amt', 'city_pop', TARGET, and engineered features
    ]
    # Make sure drop cols exist and are not in FEATURES or TARGET
    DROP_COLS = [c for c in DROP_COLS if c in all_cols and c != TARGET and c not in FEATURES]

    # Ensure TARGET exists
    if TARGET not in all_cols:
        st.error(f"Target column '{TARGET}' not found in the dataset!")
        data = None # Invalidate data

    # Check for missing features after filtering
    if not NUMERICAL_FEATURES and not CATEGORICAL_FEATURES:
         st.error("No valid features selected or found after feature engineering!")
         data = None
    else:
        st.write("Selected Numerical Features:", NUMERICAL_FEATURES)
        st.write("Selected Categorical Features:", CATEGORICAL_FEATURES)
        st.write("Final Features for Model:", FEATURES)
        st.write("Columns to Drop:", DROP_COLS)


# --- Helper Functions ---
def create_preprocessor(numerical_features, categorical_features):
    """Creates a ColumnTransformer for preprocessing."""
    transformers = []
    if numerical_features:
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        transformers.append(('num', numerical_transformer, numerical_features))

    if categorical_features:
        # Using OneHotEncoder - suitable for Random Forest, might struggle with high cardinality
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    if not transformers:
         raise ValueError("No features selected for preprocessing!")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' # Keep other columns if any (shouldn't be if FEATURES is defined correctly)
    )
    return preprocessor

def plot_confusion_matrix(cm, classes):
    """Plots a confusion matrix using seaborn."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    return fig

def plot_precision_recall_curve(y_true, y_scores):
    """Plots the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    return fig

# --- Streamlit App Layout ---
st.title("🛡️ Transaction Fraud Detection System")
st.write("""
Welcome to the Fraud Detection System. This app analyzes transaction data to identify potentially fraudulent activities.
Use the sidebar to navigate between sections.
""")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ["Data Overview", "Model Training", "Make Prediction"]
)

# --- Main App Logic ---
if data is None:
    st.error("Dataset could not be loaded or processed correctly. Please check the file path, file integrity, and feature engineering steps.")
    st.stop() # Stop execution if data isn't loaded/processed

# --- 1. Data Overview Section ---
if app_mode == "Data Overview":
    st.header("📊 Data Overview")
    st.write("Sample of Processed Data (with engineered features):")
    st.dataframe(data.head())

    st.subheader("Dataset Information")
    st.write(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    st.subheader("Basic Statistics (Numerical Features)")
    # Show stats only for the features we plan to use + target
    display_cols = NUMERICAL_FEATURES + ([TARGET] if TARGET in data.columns else [])
    if display_cols:
        st.dataframe(data[display_cols].describe())
    else:
        st.warning("No numerical features selected or available to describe.")

    st.subheader(f"Fraud Distribution ('{TARGET}')")
    if TARGET in data.columns:
        fraud_counts = data[TARGET].value_counts()
        st.write(fraud_counts)
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.countplot(x=TARGET, data=data, ax=ax_dist, palette="viridis")
        ax_dist.set_title(f"Distribution of Fraud Target (0: Not Fraud, 1: Fraud)")
        ax_dist.set_xlabel("Fraud Status")
        ax_dist.set_ylabel("Count")
        # Set custom labels for x-axis ticks
        ax_dist.set_xticks([0, 1])
        ax_dist.set_xticklabels(['Not Fraud (0)', 'Fraud (1)'])
        st.pyplot(fig_dist)
    else:
        st.warning(f"Target column '{TARGET}' not found for distribution plot.")

    st.subheader("Feature Distributions (Sample)")
    if NUMERICAL_FEATURES:
        selected_feature = st.selectbox("Select a numerical feature to visualize:", NUMERICAL_FEATURES)
        if selected_feature and TARGET in data.columns:
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.histplot(data=data, x=selected_feature, kde=True, hue=TARGET, palette="muted", ax=ax_hist)
            ax_hist.set_title(f"Distribution of {selected_feature.replace('_', ' ').title()} by Fraud Status")
            st.pyplot(fig_hist)
        elif not TARGET in data.columns:
             st.warning(f"Target column '{TARGET}' needed for hue.")
        else:
             st.warning(f"Selected feature '{selected_feature}' not found.")

    else:
        st.write("No numerical features available for distribution plots.")


    st.subheader("Correlation Heatmap (Numerical Features)")
    if NUMERICAL_FEATURES and TARGET in data.columns:
        numerical_data_for_corr = data[NUMERICAL_FEATURES + [TARGET]].copy()
        # Handle potential infinite values resulting from calculations (like distance)
        numerical_data_for_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Optionally fill NaNs if any remain, e.g., with median, though earlier steps should handle it
        # numerical_data_for_corr.fillna(numerical_data_for_corr.median(), inplace=True)

        if numerical_data_for_corr.isnull().any().any():
             st.warning("NaNs/Infs detected in data for correlation. Attempting to calculate anyway.")

        corr = numerical_data_for_corr.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8)) # Adjusted size for more features
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr, annot_kws={"size": 8})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        ax_corr.set_title("Correlation Matrix of Numerical Features and Target")
        st.pyplot(fig_corr)
    else:
        st.write("Cannot generate correlation heatmap. Need numerical features and the target column.")


# --- 2. Model Training Section ---
elif app_mode == "Model Training":
    st.header("🤖 Model Training")
    st.write("Train a Random Forest Classifier to detect fraudulent transactions.")

    # Training parameters
    test_size = st.slider("Test Set Size (%)", 10, 50, 25, 5) / 100.0
    random_state = st.number_input("Random State for Splitting", value=42, step=1)
    n_estimators = st.slider("Number of Trees (n_estimators)", min_value=50, max_value=300, value=100, step=10)
    max_depth = st.slider("Max Depth of Trees (0 for None)", min_value=0, max_value=30, value=10, step=1) # Start with limited depth
    max_depth_val = None if max_depth == 0 else max_depth

    if st.button("Train Fraud Detection Model", type="primary"):
        st.write("Starting model training...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 1. Prepare Data for Modeling
            status_text.text("Preparing data...")
            # Drop specified columns and handle potential NaNs in features/target *before* splitting
            # Although NaNs were handled in feature engineering, double-check features
            X_raw = data[FEATURES].copy()
            y = data[TARGET].copy()

            # Final check for NaNs that might have slipped through or resulted from failed FE
            if X_raw.isnull().any().any():
                 st.warning(f"NaNs detected in features before training: {X_raw.isnull().sum().sum()}. Attempting to fill with median/mode.")
                 # Fill numerical NaNs with median
                 for col in X_raw.select_dtypes(include=np.number).columns:
                     if X_raw[col].isnull().any():
                         X_raw[col].fillna(X_raw[col].median(), inplace=True)
                 # Fill categorical NaNs with mode
                 for col in X_raw.select_dtypes(include='object').columns:
                      if X_raw[col].isnull().any():
                         X_raw[col].fillna(X_raw[col].mode()[0], inplace=True) # Use mode()[0] in case of multiple modes

            if y.isnull().any():
                 st.error(f"Target column '{TARGET}' contains NaN values. Cannot train model. Please clean the data.")
                 raise ValueError(f"NaNs in target column '{TARGET}'")


            X_train, X_test, y_train, y_test = train_test_split(
                X_raw, y, test_size=test_size, random_state=random_state, stratify=y # Stratify is important for imbalanced fraud data
            )
            progress_bar.progress(20)
            st.write(f"Training data shape: {X_train.shape}")
            st.write(f"Test data shape: {X_test.shape}")


            # 2. Create Preprocessor and Model Pipeline
            status_text.text("Building preprocessing and model pipeline...")
            preprocessor = create_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth_val,
                    random_state=random_state,
                    class_weight='balanced', # Crucial for imbalanced fraud datasets
                    n_jobs=-1, # Use all available CPU cores
                    min_samples_split=10, # Add some regularization
                    min_samples_leaf=5
                ))
            ])
            progress_bar.progress(40)

            # 3. Train the Model
            status_text.text(f"Training Random Forest with {n_estimators} trees (max_depth={max_depth_val or 'None'})...")
            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            progress_bar.progress(80)

            # 4. Evaluate the Model
            status_text.text("Evaluating model...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of fraud

            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['Not Fraud (0)', 'Fraud (1)'])
            cm = confusion_matrix(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            progress_bar.progress(100)
            status_text.success(f"Model training completed in {training_time:.2f} seconds!")

            # 5. Display Results
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.caption("Note: Accuracy can be misleading in imbalanced datasets.")
            with col2:
                 st.metric("ROC AUC Score", f"{roc_auc:.4f}")
                 st.caption("Area Under the ROC Curve - better for imbalance.")
            with col3:
                 st.metric("Training Time", f"{training_time:.2f} s")


            st.subheader("Classification Report")
            st.text(report)
            st.caption("Focus on Precision, Recall, and F1-score for the 'Fraud (1)' class.")


            col_cm, col_pr = st.columns(2)
            with col_cm:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
                st.pyplot(fig_cm)

            with col_pr:
                st.subheader("Precision-Recall Curve")
                fig_pr = plot_precision_recall_curve(y_test, y_pred_proba)
                st.pyplot(fig_pr)


            # Store model in session state for prediction
            st.session_state['fraud_model'] = model
            st.session_state['features_used'] = FEATURES # Store feature list used
            st.session_state['numerical_features'] = NUMERICAL_FEATURES
            st.session_state['categorical_features'] = CATEGORICAL_FEATURES
            st.session_state['original_data_cols'] = data.columns # Store for input mapping if needed
            st.session_state['data_for_inputs'] = data # Store data to get unique values/ranges

            st.success("Model trained and saved for making predictions.")

        except ValueError as ve:
             st.error(f"Configuration Error: {ve}")
             status_text.error("Training failed due to configuration.")
             progress_bar.empty()
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
            status_text.error("Training failed.")
            progress_bar.empty() # Clear progress bar on error

# --- 3. Make Prediction Section ---
elif app_mode == "Make Prediction":
    st.header("🔮 Predict Transaction Fraud Risk")

    if 'fraud_model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section.")
    else:
        st.write("Enter the transaction details:")

        # Retrieve the trained model and features
        model = st.session_state['fraud_model']
        features_used = st.session_state['features_used']
        numerical_features = st.session_state['numerical_features']
        categorical_features = st.session_state['categorical_features']
        data_for_inputs = st.session_state['data_for_inputs'] # Get original data for ranges/options

        # Create input fields dynamically based on FEATURES
        input_data = {}
        cols = st.columns(3) # Create columns for inputs

        current_col_idx = 0
        for feature in features_used:
            target_col = cols[current_col_idx % 3] # Cycle through columns
            with target_col:
                feature_label = feature.replace('_', ' ').title()
                if feature in categorical_features:
                    unique_values = list(data_for_inputs[feature].unique())
                    # Handle potential NaNs in unique values if they exist
                    unique_values = [val for val in unique_values if pd.notna(val)]
                    # Provide a default or handle empty list case
                    default_cat_index = 0
                    if not unique_values:
                         st.warning(f"No unique values found for categorical feature: {feature}. Using placeholder.")
                         unique_values = ["N/A"]
                    elif unique_values[0] not in unique_values : # Check if default exists
                         # Select the first valid option if default isn't available
                         default_cat_index=0

                    input_data[feature] = st.selectbox(
                        f"{feature_label}:",
                        options=unique_values,
                        key=f"input_{feature}",
                        index=default_cat_index
                    )

                elif feature in numerical_features:
                    try:
                        min_val = float(data_for_inputs[feature].min())
                        max_val = float(data_for_inputs[feature].max())
                        mean_val = float(data_for_inputs[feature].mean())
                        # Define a reasonable step, avoid zero step if min=max
                        step_val = (max_val - min_val) / 100 if (max_val - min_val) > 0 else 1.0
                        if step_val == 0: step_val = 1.0 # Fallback step
                    except Exception as e: # Fallback if stats fail
                        st.warning(f"Could not get stats for {feature}: {e}. Using default range.")
                        min_val = 0.0
                        max_val = data_for_inputs[feature].max() * 1.5 if pd.notna(data_for_inputs[feature].max()) else 1000.0 # Default max
                        mean_val = data_for_inputs[feature].median() if pd.notna(data_for_inputs[feature].median()) else (min_val + max_val) / 2.0
                        step_val = (max_val-min_val)/100 or 1.0

                    input_data[feature] = st.number_input(
                        f"{feature_label}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step= step_val,
                        key=f"input_{feature}",
                        format="%.4f" # Show more decimal places for amounts/coords etc.
                    )
                else:
                     # This case shouldn't happen if FEATURES = CAT + NUM
                     st.warning(f"Feature '{feature}' not found in categorical or numerical lists.")

            current_col_idx += 1


        if st.button("Predict Fraud Risk", type="primary"):
            # Create a DataFrame from the user inputs with the correct column order
            input_df = pd.DataFrame([input_data])
            input_df = input_df[features_used] # Ensure column order matches training

            st.write("Input Data for Prediction:")
            st.dataframe(input_df)

            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]

                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.error("🚨 Alert: High risk of FRAUD detected for this transaction!")
                    st.metric("Fraud Probability", f"{prediction_proba[1]:.4f}")
                    # st.write(f"Probability of Not Fraud: {prediction_proba[0]:.4f}")
                else:
                    st.success("✅ Status: Transaction appears to be legitimate.")
                    st.metric("Fraud Probability", f"{prediction_proba[1]:.4f}")
                    # st.write(f"Probability of Not Fraud: {prediction_proba[0]:.4f}")

                # --- Logging Simulation ---
                st.subheader("Logging Prediction (Simulation)")
                log_entry = {
                    'timestamp': pd.Timestamp.now(),
                    'inputs': input_data,
                    'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
                    'probability_fraud': prediction_proba[1]
                    # 'user_id': st.session_state.user_id # Would come from login state
                }
                st.json(log_entry) # Display what would be logged
                st.info("In a full application, this prediction data would be stored in a backend database, potentially triggering alerts or reviews.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Ensure all input fields have valid values.")

# --- Footer (Optional) ---
st.sidebar.markdown("---")
st.sidebar.info("Fraud Detection System using Streamlit.")
# --- END OF FILE app.py ---
