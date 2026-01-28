import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- NETWORK INTRUSION MODEL ----------
network_df = pd.read_csv("demo_network.csv")

X_net = network_df.drop("label", axis=1)
y_net = network_df["label"]

scaler_net = StandardScaler()
X_net_scaled = scaler_net.fit_transform(X_net)

X_train, X_test, y_train, y_test = train_test_split(
    X_net_scaled, y_net, test_size=0.2, random_state=42
)

network_model = RandomForestClassifier(n_estimators=100, random_state=42)
network_model.fit(X_train, y_train)

joblib.dump(network_model, "models/network_model.pkl")
joblib.dump(scaler_net, "models/network_scaler.pkl")

print("✅ Network intrusion model trained and saved")

# ---------- WEB INTRUSION MODEL ----------
web_df = pd.read_csv("demo_web.csv")

X_web = web_df.drop("label", axis=1)
y_web = web_df["label"]

scaler_web = StandardScaler()
X_web_scaled = scaler_web.fit_transform(X_web)

X_train, X_test, y_train, y_test = train_test_split(
    X_web_scaled, y_web, test_size=0.2, random_state=42
)

web_model = RandomForestClassifier(n_estimators=100, random_state=42)
web_model.fit(X_train, y_train)

joblib.dump(web_model, "models/web_model.pkl")
joblib.dump(scaler_web, "models/web_scaler.pkl")

print("✅ Web intrusion model trained and saved")