import pandas as pd

def load_data(filepath):
    """
    Load software development data from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file containing the data.

    Returns:
        dict: Dictionary with keys 'code_churn', 'test_coverage', and 'defects',
              each mapping to a NumPy array of the corresponding column values.
    """
    df = pd.read_csv(filepath)
    return {
        'code_churn': df['code_churn'].values,
        'test_coverage': df['test_coverage'].values,
        'defects': df['defects'].values
    }
