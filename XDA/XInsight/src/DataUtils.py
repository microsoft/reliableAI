import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
from typing import Tuple, Dict, Any
import logging

def convert_cells(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Any]]]:
    """
    Convert DataFrame columns based on their data types.

    - String columns are converted to class indices.
    - Numeric columns are kept as is.
    - Datetime columns are dropped with a logged warning.
    
    Args:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The converted DataFrame.
    - dict: Mapping of column names to another dictionary of index to original value.
    """
    # Clone the input DataFrame to avoid modifying the original
    df_new = df.copy()
    col_mapping = {}

    for col in df.columns:
        col_type = df[col].dtype

        # If the column is of type 'object' (indicative of string type in DataFrame)
        if col_type == 'object':
            df_new[col], unique_vals = pd.factorize(df[col])
            col_mapping[col] = {index: value for index, value in enumerate(unique_vals)}

        # If the column is of datetime type
        elif np.issubdtype(col_type, np.datetime64):
            logging.warning(f"Dropping datetime column: {col}")
            df_new.drop(col, axis=1, inplace=True)

    return df_new, col_mapping

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    args.add_argument("--data_output", type=str, required=True, help="Path to output CSV file")
    args.add_argument("--mapping_output", type=str, required=True, help="Path to output CSV file")
    args = args.parse_args()

    df = pd.read_csv(args.input)
    df, col_mapping = convert_cells(df)
    df.to_csv(args.data_output, index=False)
    # Save the column mapping as a JSON file
    with open(args.mapping_output, 'w') as f:
        json.dump(col_mapping, f)