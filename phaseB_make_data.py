import os
from collections import Counter
from env.logistics_env import LogisticsInstance
from data.perception_synth import generate_dataset
import pandas as pd
import shutil

def make_balanced_dataset(
    out_dir='perception_data',
    samples_per_customer=200,
    size=16,
    seed=42
):
    print("\n=== Generating raw dataset ===")

    inst = LogisticsInstance(
        n_customers=10,
        seed=seed,
        max_demand=3,
        vehicle_count=3,
        vehicle_capacity=10,
        time_window_slack=40,
        tightness=0.8
    )

    raw_dir = out_dir + "_raw"
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)

    generate_dataset(inst, out_dir=raw_dir, samples_per_customer=samples_per_customer, size=size)

    labels_file = os.path.join(raw_dir, 'labels.csv')
    df = pd.read_csv(labels_file)

    print("Raw class counts:", Counter(df['label']))

    desired_counts = {
        0: samples_per_customer,
        1: samples_per_customer,
        2: samples_per_customer,
        3: samples_per_customer
    }

    balanced_rows = []
    for cls, desired_n in desired_counts.items():
        subset = df[df['label'] == cls]
        if len(subset) == 0:
            continue
        if len(subset) >= desired_n:
            subset = subset.sample(desired_n, random_state=seed)
        else:
            repeat_factor = desired_n // len(subset) + 1
            subset = pd.concat([subset] * repeat_factor).sample(desired_n, random_state=seed)

        balanced_rows.append(subset)

    balanced_df = pd.concat(balanced_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

    print("Balanced class counts:", Counter(balanced_df['label']))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)
    os.makedirs(os.path.join(out_dir, "images"))

    raw_img_dir = os.path.join(raw_dir, "images")
    final_img_dir = os.path.join(out_dir, "images")

    for _, row in balanced_df.iterrows():
        src = os.path.join(raw_img_dir, row['filename'])
        dst = os.path.join(final_img_dir, row['filename'])
        shutil.copy(src, dst)

    balanced_df.to_csv(os.path.join(out_dir, 'labels.csv'), index=False)

    print("\n=== Balanced dataset created successfully ===")
    print("Final class counts:", Counter(balanced_df['label']))
    print("Dataset directory:", os.path.abspath(out_dir))


if __name__ == "__main__":
    make_balanced_dataset()
