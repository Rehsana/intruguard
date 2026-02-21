import pandas as pd
import numpy as np
import random
from datetime import datetime

print("🔄 Starting dataset expansion and restructuring...\n")

# Define the 41 columns matching train.csv.csv and test.csv.csv structure
columns = [
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

protocol_types = ["tcp", "udp", "icmp"]
services = ["ftp_data", "ftp", "ssh", "telnet", "smtp", "http", "private", "other", "eco_i"]
flags = ["SF", "S0", "REJ", "RSTO", "SH", "S1", "S2", "S3"]
labels = ["normal", "neptune", "mscan", "saint", "smurf", "portsweep", "nmap"]

def generate_network_record(label_type):
    """Generate a single network intrusion record"""
    
    if label_type == "normal":
        duration = random.randint(0, 100)
        src_bytes = random.randint(50, 500)
        dst_bytes = random.randint(0, 5000)
        wrong_fragment = 0
        urgent = 0
        hot = random.randint(0, 2)
        num_failed_logins = 0
        logged_in = 1 if random.random() > 0.3 else 0
        num_compromised = 0
        root_shell = 0
        su_attempted = 0
        num_root = 0
        num_shells = 0
        count = random.randint(1, 20)
        srv_count = random.randint(1, 20)
        serror_rate = round(random.uniform(0, 0.5), 2)
        srv_serror_rate = round(random.uniform(0, 0.5), 2)
        flag = "SF"
    else:  # malicious
        duration = random.randint(0, 500)
        src_bytes = random.randint(0, 2000)
        dst_bytes = random.randint(0, 10000)
        wrong_fragment = random.randint(0, 1)
        urgent = random.randint(0, 1)
        hot = random.randint(1, 10)
        num_failed_logins = random.randint(1, 20)
        logged_in = 0
        num_compromised = random.randint(0, 5)
        root_shell = random.randint(0, 1)
        su_attempted = random.randint(0, 1)
        num_root = random.randint(0, 10)
        num_shells = random.randint(0, 5)
        count = random.randint(10, 500)
        srv_count = random.randint(5, 200)
        serror_rate = round(random.uniform(0.5, 1.0), 2)
        srv_serror_rate = round(random.uniform(0.5, 1.0), 2)
        flag = random.choice(["S0", "REJ", "RSTO", "S1", "S2"])
    
    # Common fields
    protocol_type = random.choice(protocol_types)
    service = random.choice(services)
    land = 0
    num_file_creations = random.randint(0, 3)
    num_access_files = random.randint(0, 2)
    num_outbound_cmds = 0
    is_host_login = 0
    is_guest_login = 0
    rerror_rate = round(random.uniform(0, 1), 2)
    srv_rerror_rate = round(random.uniform(0, 1), 2)
    same_srv_rate = round(random.uniform(0, 1), 2)
    diff_srv_rate = round(random.uniform(0, 1), 2)
    srv_diff_host_rate = round(random.uniform(0, 1), 2)
    dst_host_count = random.randint(1, 255)
    dst_host_srv_count = random.randint(1, 255)
    dst_host_same_srv_rate = round(random.uniform(0, 1), 2)
    dst_host_diff_srv_rate = round(random.uniform(0, 1), 2)
    dst_host_same_src_port_rate = round(random.uniform(0, 1), 2)
    dst_host_srv_diff_host_rate = round(random.uniform(0, 1), 2)
    dst_host_serror_rate = round(random.uniform(0, 1), 2)
    dst_host_srv_serror_rate = round(random.uniform(0, 1), 2)
    dst_host_rerror_rate = round(random.uniform(0, 1), 2)
    dst_host_srv_rerror_rate = round(random.uniform(0, 1), 2)
    
    return [
        duration, protocol_type, service, flag, src_bytes, dst_bytes,
        land, wrong_fragment, urgent, hot, num_failed_logins, logged_in,
        num_compromised, root_shell, su_attempted, num_root, num_file_creations,
        num_shells, num_access_files, num_outbound_cmds, is_host_login,
        is_guest_login, count, srv_count, serror_rate, srv_serror_rate,
        rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate,
        srv_diff_host_rate, dst_host_count, dst_host_srv_count,
        dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate,
        dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate,
        dst_host_rerror_rate, dst_host_srv_rerror_rate, label_type
    ]

# Generate expanded network demo dataset (10,000 rows)
print("📊 Generating expanded demo_network.csv (10,000 rows)...")
network_data = []
num_network_records = 10000
normal_ratio = 0.7  # 70% normal, 30% attacks

for _ in range(int(num_network_records * normal_ratio)):
    network_data.append(generate_network_record("normal"))

attack_labels = ["neptune", "mscan", "saint", "smurf"]
for _ in range(int(num_network_records * (1 - normal_ratio))):
    network_data.append(generate_network_record(random.choice(attack_labels)))

network_df = pd.DataFrame(network_data, columns=columns)
network_df.to_csv("demo_network.csv", index=False)
print(f"✅ Created demo_network.csv with {len(network_df)} rows\n")

# Generate expanded web demo dataset (10,000 rows with all 41 columns)
print("📊 Generating expanded demo_web.csv (10,000 rows)...")
web_data = []
num_web_records = 10000

for _ in range(int(num_web_records * normal_ratio)):
    # Normal web traffic
    duration = random.randint(0, 50)
    src_bytes = random.randint(50, 500)
    dst_bytes = random.randint(100, 3000)
    flag = "SF"
    label_type = "normal"
    
    web_data.append([
        duration, "tcp", "http", flag, src_bytes, dst_bytes,
        0, 0, 0, random.randint(0, 1), 0, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0,
        0, random.randint(1, 10), random.randint(1, 10), 0.0, 0.0,
        0.0, 0.0, random.uniform(0.8, 1.0), random.uniform(0, 0.2),
        0.0, random.randint(1, 50), random.randint(1, 50),
        round(random.uniform(0.7, 1.0), 2), round(random.uniform(0, 0.3), 2), round(random.uniform(0.7, 1.0), 2),
        round(random.uniform(0, 0.3), 2), 0.0, 0.0,
        0.0, 0.0, label_type
    ])

# Malicious web traffic
for _ in range(int(num_web_records * (1 - normal_ratio))):
    duration = random.randint(0, 300)
    src_bytes = random.randint(0, 5000)
    dst_bytes = random.randint(0, 50000)
    flag = random.choice(["SF", "REJ", "S0"])
    label_type = random.choice(["neptune", "portsweep", "nmap"])
    
    web_data.append([
        duration, "tcp", random.choice(["http", "smtp", "ftp"]), flag, src_bytes, dst_bytes,
        0, random.randint(0, 1), random.randint(0, 1), random.randint(0, 10), random.randint(0, 5), random.randint(0, 1),
        random.randint(0, 3), random.randint(0, 1), random.randint(0, 1), random.randint(0, 5), random.randint(0, 2),
        random.randint(0, 3), random.randint(0, 2), 0, 0,
        0, random.randint(50, 500), random.randint(50, 500), round(random.uniform(0.3, 1.0), 2), round(random.uniform(0.3, 1.0), 2),
        round(random.uniform(0, 1.0), 2), round(random.uniform(0, 1.0), 2), round(random.uniform(0, 0.5), 2), round(random.uniform(0.5, 1.0), 2),
        round(random.uniform(0, 1.0), 2), random.randint(50, 255), random.randint(50, 255),
        round(random.uniform(0, 1.0), 2), round(random.uniform(0, 1.0), 2), round(random.uniform(0, 1.0), 2),
        round(random.uniform(0, 1.0), 2), round(random.uniform(0.3, 1.0), 2), round(random.uniform(0.3, 1.0), 2),
        round(random.uniform(0, 1.0), 2), round(random.uniform(0, 1.0), 2), label_type
    ])

web_df = pd.DataFrame(web_data, columns=columns)
web_df.to_csv("demo_web.csv", index=False)
print(f"✅ Created demo_web.csv with {len(web_df)} rows\n")

# Create copies in uploads folder
print("📁 Creating copies in uploads folder...")
network_df.to_csv("uploads/demo_network.csv", index=False)
web_df.to_csv("uploads/demo_web.csv", index=False)
print("✅ Created uploads/demo_network.csv")
print("✅ Created uploads/demo_web.csv\n")

print("=" * 60)
print("🎉 Dataset expansion complete!")
print("=" * 60)
print(f"📈 Summary:")
print(f"   • demo_network.csv: {len(network_df)} rows, {len(columns)} columns")
print(f"   • demo_web.csv: {len(web_df)} rows, {len(columns)} columns")
print(f"   • All columns now match train.csv.csv structure")
print(f"   • Normal vs Attack ratio: ~70% vs ~30%")
