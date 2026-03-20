import os
import pandas as pd

df = pd.read_csv("data/train/train.csv")

missing = []

for img in df['image_id']:
    if not os.path.exists(f"data/train/images/{img}"):
        missing.append(img)

print("Missing images:", len(missing))
print(df['label'].value_counts())