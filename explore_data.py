"""
Explore the structure of the Payload-Byte CSV files.
We need to identify:
  1. Column names (especially payload byte columns and label columns)
  2. Number of rows
  3. Data types
  4. Label distribution (to compare against the paper's Figures 3 and 4)
"""
import pandas as pd
import sys

def explore_file(filepath, name):
    print(f"\n{'='*60}")
    print(f"Exploring: {name}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    # Read just the first few rows to get structure
    df_head = pd.read_csv(filepath, nrows=5)
    print(f"\nColumns ({len(df_head.columns)} total):")
    print(df_head.columns.tolist())
    print(f"\nFirst 3 rows:")
    print(df_head.head(3).to_string())
    print(f"\nData types:")
    print(df_head.dtypes)

    # Now get label distribution — read only the label column(s)
    # We need to figure out which column is the label
    # Common names: 'label', 'Label', 'attack_cat', 'category'
    # Check the last few columns which are likely labels
    print(f"\nLast 10 columns: {df_head.columns[-10:].tolist()}")
    print(f"\nFirst 5 columns: {df_head.columns[:5].tolist()}")

    # Get row count without loading full file
    # For the large CICIDS file, we count lines
    print(f"\nCounting rows (this may take a moment for large files)...")
    row_count = sum(1 for _ in open(filepath, encoding='utf-8', errors='replace')) - 1  # subtract header
    print(f"Total rows: {row_count:,}")

    # Read just the label column(s) for distribution
    # Try reading last 4 columns only for efficiency
    cols = df_head.columns.tolist()
    label_candidates = cols[-4:]  # last 4 columns likely contain labels
    print(f"\nReading label candidate columns: {label_candidates}")
    df_labels = pd.read_csv(filepath, usecols=label_candidates)
    for col in label_candidates:
        nunique = df_labels[col].nunique()
        if nunique < 50:  # likely a label/category column
            print(f"\n  Column '{col}' — {nunique} unique values:")
            dist = df_labels[col].value_counts()
            for val, count in dist.items():
                pct = count / len(df_labels) * 100
                print(f"    {val}: {count:,} ({pct:.2f}%)")

    return df_head.columns.tolist()

# Explore UNSW (smaller file first)
unsw_cols = explore_file(r"C:\TDA\data\Payload_data_UNSW.csv", "UNSW-NB15")

# Explore CICIDS
cicids_cols = explore_file(r"C:\TDA\data\Payload_data_CICIDS2017.csv", "CICIDS2017")
