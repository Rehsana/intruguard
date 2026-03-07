
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "intru_guard_secret"

import joblib

# Load trained demo models and label encoders
network_model = joblib.load("models/network_model.pkl")
network_le = joblib.load("models/network_label_encoders.pkl")

web_model = joblib.load("models/web_model.pkl")
web_le = joblib.load("models/web_label_encoders.pkl")

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

        # Fix: Save as a temp file to avoid overwriting the source file if selected from 'uploads/'
        # This prevents the browser "ERR_UPLOAD_FILE_CHANGED" error.
        filepath = os.path.join(upload_folder, f"temp_{mode}_input.csv")
        
        # Robustness: Remove existing temp file if it exists to ensure clean state
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass # Ignore if locked, save will likely fail/overwrite anyway

        file.save(filepath)

        flash("CSV file uploaded successfully", "success")

        # 3. Read CSV
        try:
            # Try reading with header first
            df = pd.read_csv(filepath)
            
            # --- AUTO-FIX: HEADERLESS NSL-KDD DATASET ---
            # NSL-KDD typically has 41, 42, or 43 columns.
            # If columns are weird (e.g. integers as strings '0', '1') or count matches NSL-KDD but names don't match
            nsl_probed_columns = [
                "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
            ]
            
            # Check if likely headerless original NSL-KDD (approx 41-43 cols)
            if 40 <= df.shape[1] <= 44:
                # If first column is NOT 'duration', assume headerless and reload/set header
                if df.columns[0] != "duration":
                    print("DEBUG: Detected headerless Network dataset. Assigning headers...")
                    df = pd.read_csv(filepath, header=None)
                    df.columns = nsl_probed_columns[:df.shape[1]]
                
            # Define features for mapping/detection (7+7 = 14 Core Features)
            network_features_a = [
                "src_bytes", "dst_bytes", "logged_in", "count", "srv_count", 
                "dst_host_srv_count", "dst_host_same_srv_rate"
            ]
            
            network_features_b = [
                "serror_rate", "srv_serror_rate", "same_srv_rate", "diff_srv_rate", 
                "dst_host_count", "dst_host_same_src_port_rate", "dst_host_serror_rate"
            ]
            
            # Full 14 features for the comprehensive demo
            network_features_full = network_features_a + network_features_b
            
            # Define Web Features (7+7 = 14 Core Features)
            web_features_a = [
                "Flow Duration", "Total Fwd Packets", "Total Backward Packets", 
                "Total Length of Fwd Packets", "Fwd Packet Length Mean", 
                "Bwd Packet Length Mean", "Flow Bytes/s"
            ]
            
            web_features_b = [
                "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Total", "Bwd IAT Total", 
                "Packet Length Mean", "Average Packet Size", "ACK Flag Count"
            ]
            
            web_features_full = web_features_a + web_features_b
            
            # --- AUTO-FIX: DATASET MAPPING ---
            # 1. NSL-KDD Legacy (41-feature)
            if mode == "network" and 40 <= df.shape[1] <= 44:
                if df.columns[0] != "duration":
                    df = pd.read_csv(filepath, header=None)
                    df.columns = nsl_probed_columns[:df.shape[1]]
                
                # Default to Set A (core model)
                orig_indices = [4, 5, 11, 22, 23, 32, 33]
                df = df.iloc[:, orig_indices].copy()
                df.columns = network_features_a
                flash("Detected legacy network dataset. Reduced to Set A features.", "info")
                
            # 2. Network Demo Full (14 features + label)
            elif mode == "network" and df.shape[1] == 15:
                df.columns = network_features_full + ["label"]
                flash("Detected 14-feature Network Demo dataset.", "info")
                
            # 3. Network Training/Testing Subset (7 features + label)
            elif mode == "network" and df.shape[1] == 8:
                if all(col in df.columns for col in network_features_a):
                    flash("Detected 7-feature Network Training set (Set A).", "info")
                elif all(col in df.columns for col in network_features_b):
                    # Set B detected! Map to Set A for the model (Demo Shift)
                    df_mapped = df[network_features_b + ["label"]].copy()
                    df_mapped.columns = network_features_a + ["label"]
                    df = df_mapped
                    flash("Detected 7-feature Network Testing set (Set B). Simulating Feature Shift...", "info")
                else:
                    df.columns = network_features_a + ["label"]
                    flash("Detected headerless 8-column network dataset. Assigned Set A headers.", "info")

            # 4. Web Demo Full (14 features + label)
            elif mode == "web" and df.shape[1] == 15:
                df.columns = web_features_full + ["label"]
                flash("Detected 14-feature Web Demo dataset.", "info")

            # 5. Web Training/Testing Subset (7 features + label)
            elif mode == "web" and df.shape[1] == 8:
                if all(col in df.columns for col in web_features_a):
                    flash("Detected 7-feature Web Training set (Set A).", "info")
                elif all(col in df.columns for col in web_features_b):
                    # Set B detected! Map to Set A for the Web model (Demo Shift)
                    df_mapped = df[web_features_b + ["label"]].copy()
                    df_mapped.columns = web_features_a + ["label"]
                    df = df_mapped
                    flash("Detected 7-feature Web Testing set (Set B). Simulating Feature Shift...", "info")
                else:
                    df.columns = web_features_a + ["label"]
                    flash("Detected headerless 8-column web dataset. Assigned Set A headers.", "info")

            elif mode == "web" and df.shape[1] == 11: # Legacy CICIDS Web 10-feature
                web_old = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Mean", "Flow Bytes/s"]
                df.columns = web_old + ["label"]
                df = df[web_features_a + ["label"]]
                flash("Detected legacy 10-feature web dataset. Migrated to Set A.", "info")

            # Set the feature set for the prediction step (always uses Set A names)
            network_features = network_features_a 
            web_features = web_features_a
            
            if df.shape[1] == 16: # Old 15-feature network
                # Map old 15 features to new 7 features
                df.columns = [
                    "src_bytes", "dst_bytes", "logged_in", "count", "srv_count", 
                    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count", 
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
                    "dst_host_serror_rate", "service", "flag", "label"
                ]
                df = df[network_features + ["label"]]
                flash("Detected old 15-feature dataset. Migrated to 7-feature set.", "info")
            elif df.shape[1] == 11: # CICIDS Web 10-feature
                df.columns = web_features + ["label"]
                flash("Detected 10-feature CICIDS Web dataset.", "info")
            elif df.shape[1] == 21: # CICIDS Web 20-feature (Feature Shift)
                # First 10 are Set A, next 10 are Set B
                full_cicids = web_features + [
                    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
                    "Fwd IAT Total", "Bwd IAT Total", "Packet Length Mean", 
                    "Packet Length Std", "Average Packet Size", "ACK Flag Count", "label"
                ]
                df.columns = full_cicids
                flash("Detected 20-feature CICIDS Web dataset (Feature Shift).", "info")
        except Exception as e:
            flash(f"CSV read error: {e}", "danger")
            return redirect(request.url)
        

        # 4. Validate empty file
        if df.empty:
            flash("Uploaded CSV is empty", "danger")
            return redirect(request.url)

        # 5. Encode and prepare data
        try:
            le_dict = network_le if mode == "network" else web_le
            
            # Define features based on mode
            all_features = network_features if mode == "network" else web_features
            
            # Check for missing features
            missing_features = [f for f in all_features if f not in df.columns]
            if missing_features:
                hint = ""
                # Check if it looks like the user uploaded the WRONG dataset type
                if mode == "web" and "duration" in df.columns:
                    hint = " (It looks like you uploaded a Network CSV to the Web module. Please switch to Network Analysis.)"
                elif mode == "network" and "request_duration" in df.columns:
                    hint = " (It looks like you uploaded a Web CSV to the Network module. Please switch to Web Analysis.)"
                elif "Flow Duration" in df.columns or "Dst Port" in df.columns:
                    hint = " (It looks like you are uploading a raw CIC-IDS2017 dataset. This demo ONLY works with the provided 'demo_network.csv' or 'demo_web.csv' files.)"
                    
                flash(f"Missing columns: {', '.join(missing_features)}.{hint}", "danger")
                return redirect(request.url)
            
            # Select and reorder columns
            df_to_process = df[all_features].copy()
            
            # Encode string columns using the same encoders used during training
            for col in all_features:
                if col in le_dict:
                    le = le_dict[col]
                    # Robust transform: map unknown labels to -1 instead of crashing (previously unseen labels error)
                    # This ensures the application stays resilient to new/unseen categorical data
                    df_to_process[col] = df_to_process[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            
            df_to_predict = df_to_process
        except Exception as e:
            flash(
                f"Data processing error: {e}",
                "danger"
            )
            return redirect(request.url)



        # 6. Predict
        if mode == "network":
            predictions = network_model.predict(df_to_predict)
        elif mode == "web":
            predictions = web_model.predict(df_to_predict)
        else:
            flash("Invalid analysis mode", "danger")
            return redirect(url_for("dashboard"))

        # 7. Convert results
        # Robust mapping: 0, "0", "normal", "benign" -> Benign. Everything else -> Attack.
        def identify_attack(p):
            s = str(p).strip().lower()
            if s in ["normal", "0", "0.0", "benign"]:
                return "Benign"
            return "Attack"
            
        df["Prediction"] = [identify_attack(p) for p in predictions]
        
        # --- ACCURACY CALCULATION ---
        accuracy_msg = None
        if "label" in df.columns:
            try:
                # Convert predictions and ground truth to comparable binary format (0: normal, 1: attack)
                def map_any_label(x):
                    s = str(x).strip().lower()
                    if s in ["normal", "0", "0.0", "benign"]:
                        return 0
                    return 1
                
                pred_binary = [map_any_label(p) for p in predictions]
                ground_truth = df["label"].apply(map_any_label)
                
                acc = accuracy_score(ground_truth, pred_binary)
                accuracy_msg = f"{acc * 100:.2f}%"
                print(f"DEBUG: Accuracy calculated for {mode} upload: {accuracy_msg}")
                print(f"DEBUG: Ground Truth Attacks: {sum(ground_truth)}, Predicted Attacks: {sum(pred_binary)}")
            except Exception as e:
                print(f"DEBUG: Accuracy calc failed: {e}")
        # ----------------------------

        # Add Severity Column for UI (High for Attack, Low for Benign)
        def get_badge(pred):
            s = str(pred).strip().lower()
            if s in ["normal", "0", "0.0", "benign"]:
                return '<span class="badge badge-success" style="font-size: 1rem; padding: 8px 12px; background-color: #00ff88; color: black;">Low</span>'
            return '<span class="badge badge-danger" style="font-size: 1rem; padding: 8px 12px;">High</span>'

        df["Severity"] = [get_badge(p) for p in predictions]

        # 8. Save result CSV
        result_path = os.path.join(upload_folder, f"result_{mode}.csv")
        # Save a clean copy without HTML
        df_clean = df.copy()
        df_clean["Severity"] = ["Low" if str(p).strip().lower() in ["normal", "0", "0.0", "benign"] else "High" for p in predictions]
        df_clean.to_csv(result_path, index=False)

        pd.set_option('display.max_colwidth', None) # Ensure full content visibility
        
        # 9. Render result page
        # Optimize: Don't render 125k rows in HTML, it crashes browser. Show top 500.
        truncated_msg = ""
        total_rows = len(df)
        
        # Calculate full counts for the cards and charts
        if "Prediction" in df.columns:
            # We use the 'Prediction' column we just created
            total_attacks = (df["Prediction"] == "Attack").sum()
            total_benign = total_rows - total_attacks
        else:
            total_attacks = 0
            total_benign = 0

        if total_rows > 500:
             truncated_msg = f" (Showing first 500 rows of {total_rows}. Download CSV for full results.)"
             flash(f"Analysis complete! {truncated_msg}", "success")
        
        return render_template(
            "result.html",
            # Fix: Limit max_rows to 500 to prevent browser crash
            tables=[df.head(500).to_html(classes="table table-striped", index=False, escape=False)],
            mode=mode,
            accuracy=accuracy_msg,  # Pass accuracy to template
            total_rows=total_rows,  # Pass real row count
            total_attacks=total_attacks,
            total_benign=total_benign
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

# --- REAL-TIME PACKET MONITORING ---
from scapy.all import sniff, IP, TCP, UDP, Ether, conf
import threading
import time
import random

# Global variable to store the latest captured packet
latest_packet_data = {
    "timestamp": time.time(),
    "src_ip": "Scanning...",
    "dst_ip": "Scanning...",
    "protocol": "WAITING",
    "length": 0,
    "prediction": "Benign"
}

def process_packet(packet):
    """Callback function for scapy sniff"""
    global latest_packet_data
    
    if IP in packet:
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        length = len(packet)
        
        # Determine protocol
        proto = "OTHER"
        if TCP in packet:
            proto = "TCP"
        elif UDP in packet:
            proto = "UDP"
        
        # Basic Heuristic for "Prediction" (Visual Demo Only)
        is_attack = "Benign"
        if length > 1200 and random.random() < 0.3:
             is_attack = "Attack"
        elif random.random() < 0.05: # Random noise
             is_attack = "Attack"
             
        latest_packet_data = {
            "timestamp": time.time(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": proto,
            "length": length,
            "prediction": is_attack
        }

def start_sniffer():
    """Background thread to sniff packets"""
    # store=False prevents keeping all packets in memory (memory leak prevention)
    try:
        # Standard Sniff (needs Npcap on Windows)
        print("Attempting standard L2 sniffing...")
        sniff(prn=process_packet, store=False)
    except Exception as e:
        print(f"L2 Sniffer Error: {e}")
        try:
            # Fallback to Layer 3 sniffing (might work without Npcap for IP packets)
            print("Attempting L3 sniffing (conf.L3socket)...")
            conf.L3socket
            sniff(prn=process_packet, store=False, iface=conf.L3socket)
        except Exception as e2:
            print(f"L3 Sniffer Error: {e2}")
            # Fallback to Simulated Data if ALL sniffing fails (e.g. no admin/npcap)
            print("Falling back to simulation mode due to sniffing failures.")
            global latest_packet_data
            protocols = ["TCP", "UDP", "HTTP", "HTTPS", "ICMP", "SSH", "FTP"]
            while True:
                time.sleep(random.uniform(0.5, 2.0))
                # update latest_packet_data manually to show *something* is broken or simulated
                packet = {
                    "timestamp": time.time(),
                    "src_ip": f"SIMULATED (No Npcap)", 
                    "dst_ip": f"10.0.0.{random.randint(1, 50)}",
                    "protocol": random.choice(protocols),
                    "length": random.randint(64, 1500),
                    "prediction": "Attack" if random.random() < 0.1 else "Benign"
                }
                latest_packet_data = packet

# Start sniffer in a background thread
# Daemon threads exit when the main program exits
sniffer_thread = threading.Thread(target=start_sniffer, daemon=True)
sniffer_thread.start()

# Live Monitor Page
@app.route("/live_monitor")
def live_monitor():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("live_monitor.html")

@app.route("/api/live_traffic")
def live_traffic_api():
    """Returns the latest captured packet"""
    return latest_packet_data

if __name__ == "__main__":
    app.run(debug=True)
