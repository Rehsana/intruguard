import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ==================== CONFIGURATION ====================

modules = {
    "network": {
        "set_a": ["src_bytes", "dst_bytes", "logged_in", "count", "srv_count", "dst_host_srv_count", "dst_host_same_srv_rate"],
        "set_b": ["serror_rate", "srv_serror_rate", "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_same_src_port_rate", "dst_host_serror_rate"],
        "train_path": "uploads/train.csv.csv",
        "test_path": "uploads/test.csv.csv",
        "model_path": "models/network_model.pkl",
        "le_path": "models/network_label_encoders.pkl"
    },
    "web": {
        "set_a": ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s"],
        "set_b": ["Flow Packets/s", "Flow IAT Mean", "Fwd IAT Total", "Bwd IAT Total", "Packet Length Mean", "Average Packet Size", "ACK Flag Count"],
        "train_path": "uploads/web_train_split.csv",
        "test_path": "uploads/web_test_split.csv",
        "model_path": "models/web_model.pkl",
        "le_path": "models/web_label_encoders.pkl"
    }
}

os.makedirs("models", exist_ok=True)

for mod_name, config in modules.items():
    print(f"\n🚀 PROCESSING MODULE: {mod_name.upper()}")
    
    try:
        # 1. Load Training Data
        print(f"LOADING: {config['train_path']}...")
        train_df = pd.read_csv(config['train_path'], header=None)
        
        # Mapping: 7 features + label
        train_df.columns = config['set_a'] + ["label"]
        print(f"DONE: Loaded {len(train_df)} rows with {train_df.shape[1]} columns.")

        # 2. Binary Label Mapping
        train_df["label_bin"] = train_df["label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
        
        X = train_df[config['set_a']]
        y = train_df["label_bin"]

        # 3. Model Training
        print(f"START: Training RandomForest (Set A)...")
        model = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        model.fit(X_train, y_train)

        # 4. Save Model
        joblib.dump(model, config['model_path'])
        joblib.dump({}, config['le_path']) 

        # 5. Feature Shift Evaluation (Test Set B)
        if os.path.exists(config['test_path']):
            print(f"TESTING: {config['test_path']} (Shifted Set B)...")
            test_df = pd.read_csv(config['test_path'], header=None)
            
            # Reset columns for test
            test_df.columns = config['set_b'] + ["label"]
            y_test_bin = test_df["label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
            
            X_test = test_df[config['set_b']]
            X_test.columns = config['set_a'] # Map B -> A for the prediction
            
            preds = model.predict(X_test)
            acc = accuracy_score(y_test_bin, preds)
            
            print(f"RESULT: Test Accuracy (Shifted): {acc * 100:.2f}%")
            if acc < 0.85:
                print(f"⚠️ WARNING: {mod_name} accuracy is below 85% requirement ({acc*100:.2f}%)")
            else:
                print(f"✅ SUCCESS: {mod_name} passed 85% requirement!")
        else:
            print(f"SKIP: {config['test_path']} not found.")

    except Exception as e:
        print(f"❌ ERROR in {mod_name}: {e}")

print("\n✨ All retrain processes completed.")
