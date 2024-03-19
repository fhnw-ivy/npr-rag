"""
This script generates a subset of data from a larger CSV file for testing or development purposes.
It helps reduce the time needed for operations like embedding generation by working with a smaller dataset.
The script accepts command-line arguments to specify the input file, the number of samples, and an optional seed for reproducibility.

Usage:
    python your_script.py <input_path> <n_samples> [--seed SEED]

Arguments:
    --input_path: Path to the input CSV file.
    --n_samples: Number of samples to include in the subset.
    --seed SEED: Seed for the random number generator to ensure reproducibility. Default is 42.
"""

import pandas as pd
import argparse
import os
import sys


def generate_subset(input_path, output_path, n, seed=42):
    """
    Generate a subset of the data for testing purposes.
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to the output CSV file.
        n (int): Number of samples to include in the subset.
        seed (int): Seed for the random number generator.
    """
    try:
        df = pd.read_csv(input_path)
        if n > len(df):
            raise ValueError(
                f"Requested number of samples ({n}) exceeds the total number of rows in the dataset ({len(df)}).")
        subset = df.sample(n, random_state=seed)
        subset.to_csv(output_path, index=False)
        print(f"Subset created successfully: {output_path}")
    except FileNotFoundError:
        print(f"File not found: {input_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Generate a subset of data from a CSV file.')
    parser.add_argument('--input_path',
                        type=str,
                        default="../data/Cleantech Media Dataset/cleantech_media_dataset_v2_2024-02-23.csv",
                        help='Path to the input CSV file.')

    parser.add_argument('--n_samples', type=int, default=102, help='Number of samples to include in the subset.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator.')

    args = parser.parse_args()

    base_path, file_name = os.path.split(args.input_path)
    file_name_without_ext, _ = os.path.splitext(file_name)
    output_path = os.path.join(base_path, f"{file_name_without_ext}_subset.csv")

    generate_subset(args.input_path, output_path, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
