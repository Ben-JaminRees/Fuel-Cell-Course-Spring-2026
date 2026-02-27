"""
data_preprocessing.py
----------------------
This module defines physical parameters, problem givens, and scrapes the assignment pdf
to find the current and voltage data for the electrolyzer/fuel cell. Data is pre-processed
to derive activation overpotentials for kinetic fitting.
"""

import pdfplumber
import pandas as pd
import numpy as np
from typing import Tuple

# Physical constants
R = 8.314  # Universal gas constant, J/(mol*K)
F = 96485  # Faraday's constant, C/mol
T = 973.0  # Temperature, K
R_ASR_CM2 = 0.067 # Ohmic resistance, ohm*cm^2
E_EQ_OX = 0.99 # Equilibrium potential for oxygen electrode reaction, V

def get_constants() -> dict:
    """
    Returns a dictionary of physical constants and parameters relevant to the problem
    with corrected units.
    
    Returns
    -------
    dict
        dict: Includes T, E_eq, and R_ASR converted to SI units (Ohm*m^2).
    """
    # Convert R_ASR from Ohm*cm^2 to Ohm*m^2 (1 m^2 = 10,000 cm^2)
    r_asr_m2 = R_ASR_CM2 / 10000.0
    return {
        "T": T,
        "E_eq": E_EQ_OX,
        "R_asr": r_asr_m2,
        "f_rt": F / (R * T) 
    }

def cheeky_assignment_table_scrape(pdf_path: str) -> pd.DataFrame:
    """
    Peels the Current (I) and Voltage (V) data from the PDF table.
    
    Inputs:
        pdf_path (str): The local path.
        
    Outputs:
        pd.DataFrame: Columns ['I_cell', 'V_cell'] containing the raw data .
    """
    data = []
    with pdfplumber.open(pdf_path) as pdf:
        # targeting the page with the table
        page = pdf.pages[0]
        table = page.extract_table()
        
        # skip headers and handle potential empty rows
        for row in table[1:]:
            if row[0] and row[1]:
                # current (A*m^-2)  and cell voltage (V) 
                data.append([float(row[0]), float(row[1])])
                
    return pd.DataFrame(data, columns=['I_cell', 'V_cell'])

def calculate_activation_overpotential(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Corrects for ohmic losses to isolate the oxygen electrode overpotential.
    
    Formula: 
    1. V_IR_free = V_cell + (I_cell * R_ASR)
    2. eta_ox = V_IR_free - E_eq 
    
    Inputs:
        df (pd.DataFrame): Raw I and V values.
        params (dict): Processed constants from get_constants().
        
    Outputs:
        pd.DataFrame: Original data plus 'V_IR_free' and 'eta_ox' columns.
    """
    # Calculate IR-free potential 
    # Note: Adding I*R works because I is negative in electrolyzer mode 
    df['V_IR_free'] = df['V_cell'] + (df['I_cell'] * params['R_asr'])
    
    # Calculate oxygen overpotential 
    df['eta_ox'] = df['V_IR_free'] - params['E_eq']
    
    # The Butler-Volmer equation often fits I_ox. 
    # I_ox = -I_cell (Fuel cell mode I > 0, O2 is cathode, so I_ox < 0) 
    df['I_ox'] = -df['I_cell']
    
    return df

