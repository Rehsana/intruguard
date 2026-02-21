import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def check_accuracy(csv_path, model_path, le_path):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    
    le_dict = joblib.load(le_path)
    for col in X.columns:
        if col in le_dict:
            X[col] = le_dict[col].transform(X[col])
            
    model = joblib.load(model_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return acc * 100

net_acc = check_accuracy("demo_network.csv", "models/network_model.pkl", "models/network_label_encoders.pkl")
web_acc = check_accuracy("demo_web.csv", "models/web_model.pkl", "models/web_label_encoders.pkl")

print(f"Network Accuracy: {net_acc:.2f}%")
print(f"Web Accuracy: {web_acc:.2f}%")
