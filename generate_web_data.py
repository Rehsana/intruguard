import pandas as pd
import random
import os

# Define the correct schema for Web Intrusion Detection
columns = ["url_length", "has_sql_keyword", "has_script", "request_size", "label"]

data = []
for _ in range(3400):
    # Randomly generate benign (0) or malicious (1) samples
    label = 0 if random.random() > 0.3 else 1
    
    if label == 0:
        # Benign traffic characteristics
        url_len = random.randint(10, 80)
        has_sql = 0
        has_script = 0
        req_size = random.randint(100, 500)
    else:
        # Malicious traffic characteristics
        url_len = random.randint(50, 500)
        has_sql = 1 if random.random() > 0.5 else 0
        has_script = 1 if random.random() > 0.5 else 0
        req_size = random.randint(500, 5000)
        
    data.append([url_len, has_sql, has_script, req_size, label])

df = pd.DataFrame(data, columns=columns)
# Save to uploads folder
df.to_csv("uploads/web_demo.csv", index=False)
print(f"Generated uploads/web_demo.csv with {len(df)} rows.")
