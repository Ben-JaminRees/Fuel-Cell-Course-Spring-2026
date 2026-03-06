import matplotlib.pyplot as plt
import numpy as np

# Force Matplotlib to use system's LaTeX compiler
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif" # Serif pairs much better with LaTeX fonts
})

def plot_tafel_analysis(j_raw: np.ndarray, v_raw: np.ndarray,
                        j_smooth: np.ndarray, v_smooth: np.ndarray,
                        best_N: int, m_best: float, b_best: float,
                        r_sq: float, i_0: float, alpha_c: float) -> None:
    """
    Generates and displays a semi-log plot of the raw data, smoothed curve,
    and regression line, including kinetics statistics in the legend.

    Args:
        j_raw (np.ndarray): Raw |J| values.
        v_raw (np.ndarray): Raw Voltage values.
        j_smooth (np.ndarray): Interpolated |J| values.
        v_smooth (np.ndarray): Interpolated Voltage values.
        best_N (int): Number of points in the regression window.
        m_best (float): Slope of the fit line.
        b_best (float): Y-intercept of the fit line.
        r_sq (float): R-squared value of the fit.
        i_0 (float): Calculated exchange current density.
        alpha_c (float): Calculated cathodic transfer coefficient.
    """
    plt.figure(figsize=(9, 7))

    # Plot data
    plt.semilogx(j_raw, v_raw, 'ko', alpha=0.3, label='Raw Data')
    plt.semilogx(j_smooth, v_smooth, 'b-', alpha=0.5, label='Smoothed Curve')

    # Generate regression line data
    log_j_fit = np.log10(j_smooth[-best_N:])
    j_fit = j_smooth[-best_N:]
    v_fit = m_best * log_j_fit + b_best

    j_start = j_fit[0]
    j_end = j_fit[-1]


    # Create detailed legend label as a single continuous string (no Python newlines)
    fit_label = (
        rf"$\left\{{ \begin{{array}}{{l}} "
        rf"\mathrm{{Fit\ }} R^2 = {r_sq:.4f} \\ "
        rf"\mathrm{{Range:\ }} {j_start:.2e} \mathrm{{\ to\ }} {j_end:.2e} \mathrm{{\ A/m^2}} \\ "
        rf"I_0 = {i_0:.2e} \mathrm{{\ A/m^2}} \\ "
        rf"\alpha_c = {alpha_c:.4f} "
        rf"\end{{array}} \right.$"
    )

# Now fit_label is surrounded by a left curly bracket on each line
    
    plt.semilogx(j_fit, v_fit, 'r--', lw=2.5, label=fit_label)

    # Formatting
    plt.xlabel('Absolute Current Density $|J|$ (A/m²)')
    plt.ylabel('Cathodic activation overpotential (V)')
    plt.title('Tafel Plot Kinetics Analysis')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tafel_analysis_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()