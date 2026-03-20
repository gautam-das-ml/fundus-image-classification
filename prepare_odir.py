import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load Excel
df = pd.read_excel("data/data.xlsx")

# Path where ALL images exist (IMPORTANT)
image_dir = "data/images"   # ← Put ALL images here


def get_label(keyword):
    keyword = str(keyword).lower()

    if 'normal' in keyword:
        return 0
    elif 'diabetic' in keyword:
        return 1
    elif 'glaucoma' in keyword:
        return 2
    elif 'macular' in keyword or 'amd' in keyword:
        return 3
    elif 'cataract' in keyword:
        return 4
    else:
        return None


data = []

for i in range(len(df)):

    # LEFT EYE
    left_img = df.iloc[i]['Left-Fundus']
    left_label = get_label(df.iloc[i]['Left-Diagnostic Keywords'])

    if left_label is not None:
        if os.path.exists(os.path.join(image_dir, left_img)):
            data.append([left_img, left_label])

    # RIGHT EYE
    right_img = df.iloc[i]['Right-Fundus']
    right_label = get_label(df.iloc[i]['Right-Diagnostic Keywords'])

    if right_label is not None:
        if os.path.exists(os.path.join(image_dir, right_img)):
            data.append([right_img, right_label])


# Create DataFrame
dataset = pd.DataFrame(data, columns=['image_id', 'label'])

print("Total samples:", len(dataset))
print(dataset['label'].value_counts())


# 🔥 Split into train/test
train_df, test_df = train_test_split(
    dataset,
    test_size=0.2,
    stratify=dataset['label'],   # VERY IMPORTANT
    random_state=42
)

# Save
#os.makedirs("data/train", exist_ok=True)
#os.makedirs("data/test", exist_ok=True)

train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))