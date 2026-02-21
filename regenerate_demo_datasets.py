import pandas as pd
import random
import numpy as np

print("🔄 Regenerating demo datasets with DISTINCT features and 50k rows...\n")

# ==================== FEATURE DEFINITIONS ====================

# Network-specific features (41 columns)
network_columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

# Web-specific features (25 columns)
web_columns = [
    "request_duration", "http_method", "user_agent_type", "url_length", "param_count",
    "special_chars_query", "content_length", "cookie_size", "referrer_type",
    "is_auth_header_present", "num_redirects", "response_code", "response_time",
    "bot_score", "ip_reputation", "geo_location_id", "session_lifetime",
    "db_query_count", "file_upload_count", "api_endpoint_id", "is_ajax",
    "header_entropy", "payload_entropy", "malicious_signatures_count", "label"
]

num_records = 50000
network_accuracy_target = 0.915 # To hit 90-93% range
web_accuracy_target = 0.905 # To hit 90-91% range

# ==================== NETWORK DATA GENERATION ====================
print(f"📊 Generating demo_network.csv ({num_records} rows)...")

network_data = []
services = ["ftp_data", "ftp", "ssh", "telnet", "smtp", "http", "private", "other"]

for i in range(num_records):
    # Determine noise
    is_noise = random.random() > network_accuracy_target
    label_is_attack = random.random() > 0.7 # 30% attack ratio
    
    # If it's NOT noise, the features match the label.
    # If it IS noise, the features are the OPPOSITE of the label.
    pattern_is_attack = label_is_attack if not is_noise else not label_is_attack

    if label_is_attack:
        label = random.choice(["neptune", "mscan", "saint", "portsweep"])
    else:
        label = "normal"

    if pattern_is_attack:
        # Strict Attack Pattern
        duration = random.randint(500, 2000)
        src_bytes = random.randint(10000, 50000)
        dst_bytes = random.randint(20000, 80000)
        count = random.randint(150, 255)
        serror_rate = 1.0
        logged_in = 0
        flag = random.choice(["S0", "REJ"])
        same_srv_rate = 0.1
        dst_host_same_srv_rate = 0.1
    else:
        # Strict Normal Pattern
        duration = random.randint(0, 40)
        src_bytes = random.randint(50, 600)
        dst_bytes = random.randint(100, 1500)
        count = random.randint(1, 10)
        serror_rate = 0.0
        logged_in = 1
        flag = "SF"
        same_srv_rate = 1.0
        dst_host_same_srv_rate = 1.0

    # Fill in other fields
    network_data.append([
        duration, random.choice(["tcp", "udp", "icmp"]), random.choice(services), flag,
        src_bytes, dst_bytes, 0, 0, 0, random.randint(0, 1), 0, logged_in,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, count, random.randint(1, 30),
        serror_rate, serror_rate, 0.0, 0.0,
        round(same_srv_rate, 2), 0.0, 0.0, 255, 255, 
        round(dst_host_same_srv_rate, 2),
        0.0, 0.5, 0.0, serror_rate, serror_rate, 0.0, 0.0,
        label
    ])

network_df = pd.DataFrame(network_data, columns=network_columns)
network_df.to_csv("demo_network.csv", index=False)
network_df.to_csv("uploads/demo_network.csv", index=False)
print(f"✅ Created demo_network.csv with {len(network_df)} rows and {len(network_columns)} columns\n")

# ==================== WEB DATA GENERATION ====================
print(f"📊 Generating demo_web.csv ({num_records} rows)...")

web_data = []
methods = ["GET", "POST", "PUT", "DELETE"]
agents = ["Chrome", "Firefox", "Safari", "Bot", "Unknown"]

for i in range(num_records):
    # Determine noise
    is_noise = random.random() > web_accuracy_target
    label_is_attack = random.random() > 0.8 # 20% attack ratio for web
    
    pattern_is_attack = label_is_attack if not is_noise else not label_is_attack

    if label_is_attack:
        label = random.choice(["sql_injection", "xss", "lfi", "rce"])
    else:
        label = "normal"

    if pattern_is_attack:
        # Strict Web Attack Pattern
        url_length = random.randint(300, 1500)
        special_chars = random.randint(20, 80)
        bot_score = 1.0
        signatures = random.randint(2, 8)
        response_code = 403
    else:
        # Strict Normal Web Pattern
        url_length = random.randint(15, 80)
        special_chars = random.randint(0, 4)
        bot_score = 0.0
        signatures = 0
        response_code = 200

    web_data.append([
        random.randint(5, 500), random.choice(methods), random.choice(agents),
        url_length, random.randint(0, 10), special_chars, random.randint(100, 10000),
        random.randint(50, 2000), random.choice(["internal", "external", "none"]),
        random.randint(0, 1), random.randint(0, 3), random.choice([200, 200, 200, 404, 500, 302]),
        random.randint(10, 1000), bot_score, round(random.uniform(0, 1), 2),
        random.randint(1, 200), random.randint(60, 3600), random.randint(0, 50),
        random.randint(0, 2), random.randint(1, 500), random.randint(0, 1),
        round(random.uniform(2, 6), 2), round(random.uniform(2, 6), 2),
        signatures, label
    ])

web_df = pd.DataFrame(web_data, columns=web_columns)
web_df.to_csv("demo_web.csv", index=False)
web_df.to_csv("uploads/demo_web.csv", index=False)
print(f"✅ Created demo_web.csv with {len(web_df)} rows and {len(web_columns)} columns\n")

print("=" * 60)
print("🎉 Dataset regeneration complete!")
print("=" * 60)
print(f"📈 Summary:")
print(f"   • demo_network.csv: {len(network_df)} rows, {len(network_columns)} features")
print(f"   • demo_web.csv:     {len(web_df)} rows, {len(web_columns)} features")
print(f"   • Network Target:   90-93% (Control noise: {100-network_accuracy_target*100:.1f}%)")
print(f"   • Web Target:       90-91% (Control noise: {100-web_accuracy_target*100:.1f}%)")
