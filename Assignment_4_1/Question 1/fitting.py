import numpy as np
from scipy.interpolate import PchipInterpolator

def smooth_data(j: np.ndarray, v: np.ndarray, num_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a smoothly interpolated curve in log-x space using PCHIP.

    Args:
        j (np.ndarray): Array of absolute current density values.
        v (np.ndarray): Array of voltage values.
        num_points (int, optional): The number of points to generate for the smooth curve. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - log_j_int: Array of linearly spaced log10(|J|) values.
            - j_int: Array of the interpolated |J| values (standard space).
            - v_int: Array of the interpolated voltage values.
    """
    log_j = np.log10(j)
    log_j_int = np.linspace(log_j.min(), log_j.max(), num_points)
    j_int = 10**log_j_int
    v_int = PchipInterpolator(log_j, v)(log_j_int)
    
    return log_j_int, j_int, v_int

def find_best_fit(log_j_int: np.ndarray, v_int: np.ndarray) -> tuple[int, float, float, float]:
    """
    Finds the optimal linear regression window using a 99% asymptote threshold algorithm.

    Args:
        log_j_int (np.ndarray): Smoothed array of log10(|J|) values.
        v_int (np.ndarray): Smoothed array of voltage values.

    Returns:
        tuple[int, float, float, float]:
            - best_N: The optimal number of points from the end of the array to include in the fit.
            - m_best: The slope of the line of best fit.
            - b_best: The y-intercept of the line of best fit.
            - r_squared: The R-squared value for the regression.
    """
    # Define slopes to find the 99% target
    m_tot = (v_int[-1] - v_int[0]) / (log_j_int[-1] - log_j_int[0])
    m_last10, _ = np.polyfit(log_j_int[-10:], v_int[-10:], 1)
    m_target = m_tot + 0.99 * (m_last10 - m_tot)
    
    # Calculate all possible fits expanding backwards
    N_vals = list(range(11, len(log_j_int)))
    fits = [np.polyfit(log_j_int[-N:], v_int[-N:], 1) for N in N_vals]
    
    # Find fit closest to target
    best_idx = int(np.argmin([abs(f[0] - m_target) for f in fits]))
    best_N = N_vals[best_idx]
    m_best, b_best = fits[best_idx]
    
    # Calculate R-squared
    y_true = v_int[-best_N:]
    y_pred = m_best * log_j_int[-best_N:] + b_best
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r_squared = float(1 - (ss_res / ss_tot))
    
    return best_N, m_best, b_best, r_squared