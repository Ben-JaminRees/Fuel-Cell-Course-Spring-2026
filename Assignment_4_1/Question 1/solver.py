def solve_kinetics(m: float, b: float, temp_k: float) -> tuple[float, float]:
    """
    Calculates exchange current density (i_0) and cathodic transfer coefficient (alpha_c)
    from the Tafel plot regression parameters.

    Args:
        m (float): The slope of the log10(|J|) vs Voltage regression line.
        b (float): The y-intercept of the log10(|J|) vs Voltage regression line.
        temp_k (float): The operating temperature in Kelvin.

    Returns:
        tuple[float, float]:
            - i_0: Exchange current density in A/m^2.
            - alpha_c: Cathodic transfer coefficient (dimensionless).
    """
    R = 8.314      # J/(mol K)
    F = 96485.0    # C/mol

    # Map regression components to Tafel equation: eta = A - B*log(-I)
    # slope (m) = -B
    # intercept (b) = A
    B_val = -m
    A_val = b
    
    # alpha_c = 2.303 * RT / (B * F)
    alpha_c = (2.303 * R * temp_k) / (B_val * F)
    
    # A = B * log10(i_0) --> log10(i_0) = A / B
    log_i0 = A_val / B_val
    i_0 = 10**log_i0
    
    return i_0, alpha_c