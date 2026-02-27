"""
fitting.py

This module defines the Butler-Volmer kinetic model and uses numerical 
optimization to fit exchange current density and transfer coefficients.
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd

def butler_volmer_equation(eta: np.ndarray, i0_scaled: float, α_a: float, α_c: float, f_rt: float) -> np.ndarray:
    """
    Calculates current density based on the Butler-Volmer model.
    
    Formula: I = i0 * [exp(α_a * F/RT * eta) - exp(-α_c * F/RT * eta)] 
    
    Inputs:
        eta (np.ndarray): Activation overpotential (V)
        i0 (float): Exchange current density (A/m^2)
        α_a (float): Anodic transfer coefficient
        α_c (float): Cathodic transfer coefficient
        f_rt (float): Pre-calculated F/(R*T) value

    Returns:
        np.ndarray: B-V equation predicted current density (A/m^2)
    """
    i0 = i0_scaled * 1000.0  # Scale back to allow more precision in optimization
    term_anodic = np.exp(α_a * f_rt * eta)
    term_cathodic = np.exp(-α_c * f_rt * eta)
    return i0 * (term_anodic - term_cathodic)

def objective_function(params: list, eta_data: np.ndarray, i_data: np.ndarray, f_rt: float) -> float:
    """
    Calculates the Sum of Squared Residuals (SSR) between experimental and model I.
    
    Inputs:
        params (list): Current guesses for [i0, α_a, α_c] 
        eta_data: Experimental overpotential
        i_data: Experimental current

    Returns:
        float: The SSR value to minimize
    """
    i0_scaled, α_a, α_c = params
    
    # Predict current using the B-V equation
    i_predicted = butler_volmer_equation(eta_data, i0_scaled, α_a, α_c, f_rt)
    
    # Calculate Residuals (Experimental - Predicted)^2 
    residuals = ((i_data - i_predicted) / 1000.0) ** 2
    ssr = np.sum(residuals)
    
    print(f"Testing: i0={i0_scaled:.4f}, aa={α_a:.4f}, ac={α_c:.4f} | Error (SSR): {ssr:.2e}")

    return ssr

def fit_kinetic_parameters(df: pd.DataFrame, f_rt: float) -> dict:
    """
    Runs the optimization solver to find the best-fit i0, α_a, and α_c.

    Inputs:
        df (pd.DataFrame): DataFrame containing 'eta_ox' and 'I_ox' columns.
        f_rt (float): Pre-calculated F/(R*T) value from get_constants().

    Returns:
        dict: Best-fit parameters and optimization success status.
    """

    global solver_history
    solver_history = []  # Reset for new run

    # 1. Initial Guesses
    # Starting with generic values; solver will iterate from here
    initial_guess = [4.0, 0.15, 0.13] 
    
    # 2. Constraints: All parameters must be >= 0 
    bounds = [(1e-6, None), (1e-6, 2.0), (1e-6, 2.0)]
    
    # 3. Optimization using GRG-style Nonlinear solver (SLSQP in SciPy) 
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(df['eta_ox'].values, df['I_ox'].values, f_rt),
        bounds=bounds,
        method='SLSQP',
        options={
        'ftol': 1e-15,  # Much tighter function tolerance (default is ~1e-7)
        'gtol': 1e-15,  # Much tighter gradient tolerance
        'maxiter': 1000 # Allow more steps if needed
        }

    )
    
    if result.success:
        i0_fit, α_a_fit, α_c_fit = result.x
        return {
            "i0": i0_fit,
            "alpha_a": α_a_fit,
            "alpha_c": α_c_fit,
            "success": True
        }
    else:
        return {"success": False, "message": result.message}