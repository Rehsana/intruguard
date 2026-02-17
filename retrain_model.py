import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Define NSL-KDD Columns
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Path to the uploaded dataset (NSL-KDD)
dataset_path = "uploads/train.csv.csv" # The user's uploaded 125k row file

print(f"🔄 Loading dataset from {dataset_path}...")

try:
    # 1. Load Data (Headerless)
    df = pd.read_csv(dataset_path, header=None)
    
    # Check if 42 columns (standard NSL-KDD) or 43 (with difficulty)
    if df.shape[1] == 42:
        df.columns = columns
    elif df.shape[1] == 43:
        df.columns = columns + ["difficulty"]
    else:
        # Fallback: Just take first 42
        df = df.iloc[:, :42]
        df.columns = columns

    print(f"✅ Loaded {len(df)} rows. Columns assigned.")

    # 2. Preprocessing
    # Drop categorical columns to avoid encoding issues in app.py (StandardScaler only handles numeric)
    # The numeric columns in NSL-KDD are sufficient for high accuracy.
    drop_cols = ["protocol_type", "service", "flag", "label"]
    if "difficulty" in df.columns:
        drop_cols.append("difficulty")
        
    X = df.drop(drop_cols, axis=1, errors='ignore')
    
    # Process Label (0 = normal, 1 = attack)
    y = df["label"].apply(lambda x: 0 if str(x).strip() == "normal" else 1)

    print(f"📊 Training on {X.shape[1]} numeric features...")

    # 3. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 5. Train Model
    print("🚀 Training Random Forest (this may take a minute)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6. Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"🏆 Model Accuracy on Test Split: {accuracy * 100:.2f}%")

    # 7. Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/network_model.pkl")
    joblib.dump(scaler, "models/network_scaler.pkl")
    print("💾 Model saved to models/network_model.pkl")

except Exception as e:
    print(f"❌ Error: {e}")
