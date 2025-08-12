import pandas as pd
import argparse

def generate_reflection_duplicates(input_file, output_file):
    """
    Reads a dataset, generates the reflection symmetry duplicates,
    and saves the new dataset including the original and mirrored entries.

    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path to save the new dataset with duplicated reflections.
    """

    # Load dataset
    df = pd.read_csv(input_file)

    # Drop the index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Define reflection symmetry mapping for 6-helicene
    reflection_indices = {
        "Pos1": "Pos16", "Pos2": "Pos15", "Pos3": "Pos14", "Pos4": "Pos13",
        "Pos5": "Pos12", "Pos6": "Pos11", "Pos7": "Pos10", "Pos8": "Pos9"
    }

    # Generate mirrored dataset
    mirrored_df = df.copy()
    for pos1, pos2 in reflection_indices.items():
        mirrored_df[pos1], mirrored_df[pos2] = df[pos2], df[pos1]

    # Append mirrored rows to original dataset
    full_dataset = pd.concat([df, mirrored_df], ignore_index=True)

    # Save the new dataset to a CSV file
    full_dataset.to_csv(output_file, index=False)
    print(f"Reflection symmetry duplicates generated and saved to {output_file}")

# Main execution for command-line usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reflection symmetry duplicates for a 6-helicene dataset.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the new dataset with reflection duplicates.")

    args = parser.parse_args()

    generate_reflection_duplicates(args.input_file, args.output_file)
