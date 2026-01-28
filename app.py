from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = "intru_guard_secret"

import joblib

# Load trained demo models
network_model = joblib.load("models/network_model.pkl")
network_scaler = joblib.load("models/network_scaler.pkl")

web_model = joblib.load("models/web_model.pkl")
web_scaler = joblib.load("models/web_scaler.pkl")

# Dummy users for login
users = {
    "admin": "admin123",
    "user": "user123"
}

# --- ROUTES ---

# Login page
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            session["user"] = username
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

# Dashboard page
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/upload/<mode>", methods=["GET", "POST"])
def upload(mode):
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        # 1. Check file presence
        if "csv_file" not in request.files:
            flash("No file part found", "danger")
            return redirect(request.url)

        file = request.files["csv_file"]

        if file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        # 2. Save file safely
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        flash("CSV file uploaded successfully", "success")

        # 3. Read CSV
        try:
            df = pd.read_csv(filepath)
            # Robustness: Strip whitespace from column names
            df.columns = df.columns.str.strip()
            print(f"DEBUG: Uploaded file columns: {df.columns.tolist()}")
        except Exception as e:
            flash(f"CSV read error: {e}", "danger")
            return redirect(request.url)
        

        # 4. Validate empty file
        if df.empty:
            flash("Uploaded CSV is empty", "danger")
            return redirect(request.url)

        # 5. Scale data
        try:
            scaler = network_scaler if mode == "network" else web_scaler
            
            # Robustly filter columns to match what the model expects
            if hasattr(scaler, "feature_names_in_"):
                required_features = list(scaler.feature_names_in_)
                
                # Check for missing features
                missing_features = [f for f in required_features if f not in df.columns]
                if missing_features:
                    hint = ""
                    # Check if it looks like the user uploaded the WRONG dataset type
                    if mode == "web" and "duration" in df.columns:
                        hint = " (It looks like you uploaded a Network CSV to the Web module. Please switch to Network Analysis.)"
                    elif mode == "network" and "url_length" in df.columns:
                        hint = " (It looks like you uploaded a Web CSV to the Network module. Please switch to Web Analysis.)"
                    elif "Flow Duration" in df.columns or "Dst Port" in df.columns:
                        hint = " (It looks like you are uploading a raw CIC-IDS2017 dataset. This demo ONLY works with the simple 'demo_network.csv' or 'demo_web.csv' provided in the project root. The models are not trained on raw CIC-IDS2017 data.)"
                        
                    flash(f"Missing columns: {', '.join(missing_features)}.{hint}", "danger")
                    return redirect(request.url)
                    
                # Filter and reorder DataFrame to match scaler's features exactly
                # This drops extra columns like 'label', 'flag', etc. automatically
                df_to_scale = df[required_features]
            else:
                # Fallback if attribute is missing (shouldn't happen with 1.8.0)
                df_to_scale = df

            df_scaled = scaler.transform(df_to_scale)
        except Exception as e:
            flash(
                f"Data processing error: {e}",
                "danger"
            )
            return redirect(request.url)

        # 6. Predict
        if mode == "network":
            predictions = network_model.predict(df_scaled)
        elif mode == "web":
            predictions = web_model.predict(df_scaled)
        else:
            flash("Invalid analysis mode", "danger")
            return redirect(url_for("dashboard"))

        # 7. Convert results
        df["Prediction"] = ["Attack" if p == 1 else "Benign" for p in predictions]
        
        # Add Severity Column for UI (High for Attack, Low for Benign)
        def get_badge(pred):
            if pred == "Attack":
                return '<span class="badge badge-danger" style="font-size: 1rem; padding: 8px 12px;">High</span>'
            return '<span class="badge badge-success" style="font-size: 1rem; padding: 8px 12px; background-color: #00ff88; color: black;">Low</span>'

        df["Severity"] = [get_badge(p) for p in df["Prediction"]]

        # 8. Save result CSV
        result_path = os.path.join(upload_folder, f"result_{mode}.csv")
        # Save a clean copy without HTML
        df_clean = df.copy()
        df_clean["Severity"] = ["High" if p == "Attack" else "Low" for p in df_clean["Prediction"]]
        df_clean.to_csv(result_path, index=False)

        # 9. Render result page
        return render_template(
            "result.html",
            tables=[df.to_html(classes="table table-striped", index=False, escape=False)],
            mode=mode
        )

    return render_template("upload.html", mode=mode)

# Download result CSV
@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join("models", filename), as_attachment=True)

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

# Live Monitor Page
@app.route("/live_monitor")
def live_monitor():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("live_monitor.html")

# API for Real-Time Traffic Simulation
@app.route("/api/live_traffic")
def live_traffic_api():
    import random
    import time
    
    protocols = ["TCP", "UDP", "HTTP", "HTTPS", "ICMP", "SSH", "FTP"]
    
    # Simulate a packet
    packet = {
        "timestamp": time.time(),
        "src_ip": f"192.168.1.{random.randint(2, 255)}",
        "dst_ip": f"10.0.0.{random.randint(1, 50)}",
        "protocol": random.choice(protocols),
        "length": random.randint(64, 1500),
        # 10% chance of Attack
        "prediction": "Attack" if random.random() < 0.1 else "Benign"
    }
    
    return packet

if __name__ == "__main__":
    app.run(debug=True)