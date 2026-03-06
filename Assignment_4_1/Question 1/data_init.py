import pandas as pd
import numpy as np

# --- Physical Constants ---
R: float = 8.314      # Universal gas constant in J/(mol K)
T: float = 298.15     # Temperature in Kelvin (25 C)
F: float = 96485.0    # Faraday's constant in C/mol

def load_and_clean_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads an Excel file, forces columns to numeric types, drops NaNs, and sorts.

    Args:
        file_path (str): The relative or absolute path to the Excel data file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 1D numpy arrays:
            - j_vals: The absolute, non-zero current densities sorted in ascending order.
            - v_vals: The corresponding voltage/overpotential values.
    """
    # Read and coerce to numeric, turning text/headers to NaN, then drop NaNs
    df = pd.read_excel(file_path).apply(pd.to_numeric, errors='coerce').dropna()
    
    # Extract absolute current density and voltage
    j = np.abs(df.iloc[:, 0].values)
    v = df.iloc[:, 1].values
    
    # Drop zeros to prevent log(0) errors later
    valid_mask = j > 0
    j, v = j[valid_mask], v[valid_mask]
    
    # Sort by current density
    sort_idx = np.argsort(j)
    
    return j[sort_idx], v[sort_idx]