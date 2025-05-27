import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return {
        'code_churn': df['code_churn'].values,
        'test_coverage': df['test_coverage'].values,
        'defects': df['defects'].values
    }
