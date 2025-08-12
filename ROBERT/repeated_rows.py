import pandas as pd

def detect_repeated_rows(file_path, output_path=None):
    """
    Detects repeated rows in a dataset and optionally saves them to a new file.

    Parameters:
    - file_path (str): Path to the CSV file containing the dataset.
    - output_path (str, optional): Path to save the duplicate rows (if provided).

    Returns:
    - A DataFrame containing the duplicate rows.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Identify duplicate rows
    duplicates = df[df.duplicated(keep=False)]  # keep=False marks all occurrences as duplicates
    
    if duplicates.empty:
        print("No repeated rows found.")
    else:
        print(f"Found {duplicates.shape[0]} repeated rows:")
        print(duplicates)

        if output_path:
            duplicates.to_csv(output_path, index=False)
            print(f"Duplicate rows saved to {output_path}")

    return duplicates

# Detect and optionally save repeated rows
detect_repeated_rows("dataset_Halogenos_Rmax.csv", "duplicates.csv")