"""
Ben Rees
02/27/2026

main.py

Main file for curve fitting. Coordinates data extraction, 
kinetic fitting, and performance evaluation.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import get_constants, cheeky_assignment_table_scrape, calculate_activation_overpotential
from fitting import fit_kinetic_parameters, butler_volmer_equation

def calculate_r_squared(y_true, y_pred):
    """Calculates the coefficient of determination."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def main():
    # 1. Setup and Data Extraction
    pdf_path = "ME5895_Asgn_Mod3_2.pdf"
    params = get_constants()
    
    print("--- 1. Extracting and Cleaning Data ---")
    raw_df = cheeky_assignment_table_scrape(pdf_path)
    df = calculate_activation_overpotential(raw_df, params)

    print(df[['I_cell', 'V_cell', 'eta_ox']])
    
    # 2. Perform the Fitting
    print("--- 2. Fitting Butler-Volmer Parameters ---")
    fit_results = fit_kinetic_parameters(df, params['f_rt'])
    
    if not fit_results['success']:
        print(f"Fitting failed: {fit_results.get('message')}")
        return

    i0, aa, ac = fit_results['i0'], fit_results['alpha_a'], fit_results['alpha_c']
    
    # 3. Evaluation: How well does it fit?
    # Predict current using the optimized parameters
    df['I_predicted'] = butler_volmer_equation(df['eta_ox'], i0, aa, ac, params['f_rt'])
    r2 = calculate_r_squared(df['I_ox'], df['I_predicted'])
    
    print("\n--- FINAL RESULTS ---")
    print(f"Exchange Current Density (i0): {i0:.2f} A/m^2")
    print(f"Anodic Transfer Coeff (alpha_a): {aa:.4f}")
    print(f"Cathodic Transfer Coeff (alpha_c): {ac:.4f}")
    print(f"Model Accuracy (R^2): {r2:.4f}")
    
    # 4. Visualization
    # Configure plot style and size
    plt.figure(figsize=(10, 7))
    
    # Use LaTeX-style formatting for axis labels and title
    # $...$ triggers the TeX parser in Matplotlib
    plt.scatter(df['eta_ox'], df['I_ox'], color='navy', label='Experimental Data (IR-corrected)')
    
    # Generate smooth B-V curve
    eta_range = np.linspace(df['eta_ox'].min(), df['eta_ox'].max(), 100)
    i_model = butler_volmer_equation(eta_range, i0, aa, ac, params['f_rt'])
    
    plt.plot(eta_range, i_model, color='crimson', lw=2, linestyle='--', 
             label=rf'Butler-Volmer Fit ($i_0$={(i0*1000.0):.1f}, $\alpha_a$={aa:.3f}, $\alpha_c$={ac:.3f}, $R^2$={r2:.4f})')

    # Axis Labels with LaTeX math
    plt.xlabel(r'Oxygen Overpotential, $\eta_{ox}$ [V]', fontsize=12)
    plt.ylabel(r'Current Density, $i_{ox}$ [A/m$^2$]', fontsize=12)
    
    # Title
    plt.title(r'Solid Oxide Fuel Cell: Oxygen Electrode Kinetic Fit', fontsize=14, fontweight='bold')
    
    plt.legend(loc='best', frameon=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.6)

    # Save the figure as a PDF in the active folder
    output_filename = "SOFC_Kinetic_Fit_Results.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    
    print(f"\n--- Visualization Saved ---")
    print(f"Graph saved as '{output_filename}' in: {os.getcwd()}")
    
    plt.show()

if __name__ == "__main__":
    main()