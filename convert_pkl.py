import pandas as pd
import numpy as np
from preprocessing import preprocess_wafer

PICKLE_PATH = r"c:/Users/manas/Desktop/Projects/Wafer/LSWMD.pkl/LSWMD.pkl"

SAMPLES_PER_DEFECT = 12000
NONE_SAMPLES = 8000

print("Loading pickle...")
df = pd.read_pickle(PICKLE_PATH)
print("Original shape:", df.shape)


# ---------------- CLEAN LABELS ----------------
def clean_label(x):
    if isinstance(x, np.ndarray) and len(x) > 0 and len(x[0]) > 0:
        return x[0][0]
    return "none"

df["clean_label"] = df["failureType"].apply(clean_label)

print("Available classes:")
print(df["clean_label"].value_counts())


# ---------------- BALANCE DATASET ----------------
print("\nBalancing dataset...")

# Separate none and defect classes
defect_df = df[df["clean_label"] != "none"]
none_df = df[df["clean_label"] == "none"]

# Sample equal amount per defect class
balanced_defects = defect_df.groupby("clean_label").apply(
    lambda x: x.sample(
        n=min(SAMPLES_PER_DEFECT, len(x)),
        random_state=42
    )
).reset_index(drop=True)

# Sample limited none
balanced_none = none_df.sample(
    n=min(NONE_SAMPLES, len(none_df)),
    random_state=42
)

# Combine
balanced_df = pd.concat([balanced_defects, balanced_none]).reset_index(drop=True)

print("Balanced dataset shape:", balanced_df.shape)
print("Balanced class distribution:")
print(balanced_df["clean_label"].value_counts())


# ---------------- CREATE MEMORY-MAPPED ARRAY ----------------
num_samples = len(balanced_df)

print("\nCreating memory-mapped dataset...")

polar_array = np.memmap(
    "polar_strips.npy",
    dtype="float32",
    mode="w+",
    shape=(num_samples, 64, 360)
)

labels_array = np.empty(num_samples, dtype=object)


# ---------------- PROCESS SAMPLES ----------------
print("Converting to normalized polar strips...")

for i in range(num_samples):
    wafer_map = balanced_df.iloc[i]["waferMap"]

    polar = preprocess_wafer(wafer_map)  # (64, 360)

    # Normalize: 0,1,2 → 0.0, 0.5, 1.0
    polar = polar.astype(np.float32) / 2.0

    polar_array[i] = polar
    labels_array[i] = balanced_df.iloc[i]["clean_label"]

    if i % 5000 == 0:
        print(f"Processed {i}/{num_samples}")

# Flush to disk
polar_array.flush()

# Save labels
pd.DataFrame({"label": labels_array}).to_csv("labels.csv", index=False)

print("\nConversion complete.")
print("Final dataset shape:", polar_array.shape)