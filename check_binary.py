import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def check_binary_accuracy(csv_path, model_path, le_path):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    
    le_dict = joblib.load(le_path)
    for col in X.columns:
        if col in le_dict:
            X[col] = le_dict[col].transform(X[col])
            
    model = joblib.load(model_path)
    predictions = model.predict(X)
    
    pred_binary = [0 if p == "normal" else 1 for p in predictions]
    ground_truth = [0 if l == "normal" else 1 for l in y]
    
    acc = accuracy_score(ground_truth, pred_binary)
    return acc * 100

net_acc = check_binary_accuracy("demo_network.csv", "models/network_model.pkl", "models/network_label_encoders.pkl")
web_acc = check_binary_accuracy("demo_web.csv", "models/web_model.pkl", "models/web_label_encoders.pkl")

print(f"Network Binary Accuracy: {net_acc:.2f}%")
print(f"Web Binary Accuracy: {web_acc:.2f}%")
