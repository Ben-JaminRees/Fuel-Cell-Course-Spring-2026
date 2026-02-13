# Ben Rees 2/12/26
# For assignment 2.2 in fuel cells
import numpy as np
import matplotlib.pyplot as plt
# take NIST data, numerically integrate to find Tds
# global defs:
# for H2+(1/2)O2->H20
N_H2 = 1.0 # mol
N_O2 = 0.5 # mol
N_H2O = 1.0 # mol
# use all units as J/mol*K, P in atm
R = 8.31446 # J/mol*K
F = 96485.0 # C/mol
n = 2.0 # electrons
T_REF = 298.15 # K
P_REF = 1.0 # atm
CONVERT_C_TO_K = 273.15 # K
T_MIN_GLOBAL = T_REF
T_MAX_GLOBAL = 800.0 + CONVERT_C_TO_K # C -> K

# species data from NIST, with different temp regimes
# for water vapor, properties are assumed to remain X(T=500)
# below 500K
SPECIES_DATA = {
    'H2O_gas': [
        {
            'range': (T_REF, 500.0),
            'coeffs': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 206.534251, -6.92120938],
            'h_form': -241826.4, # J/mol
            's_ref': 188.84 # J/mol*K
        },
        {
            'range': (500.0, 1700.0),
            'coeffs': [30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264],
            'h_form': -241830.0, # J/mol
            's_ref': 188.84 # J/mol*K
        }
    ],
    'H2': [
        {
            'range': (T_REF, 1000.0), # K
            'coeffs': [33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974, 0.0],
            'h_form': 0.0, # J/mol
            's_ref': 130.68 # J/mol*K
        },
        {
            'range': (1000.0, 2500.0), # K
            'coeffs': [18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 156.288133, 0.0],
            'h_form': 0.0, # J/mol
            's_ref': 130.68 # J/mol*K
        }
    ],
    'O2': [
        {
            'range': (T_REF, 700.0), # K
            'coeffs': [31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945, 0.0],
            'h_form': 0.0, # J/mol
            's_ref': 205.15 # J/mol * K
        },
        {
            'range': (700.0, 2000.0), # K
            'coeffs': [30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663, 0.0],
            'h_form': 0.0, # J/mol
            's_ref': 205.15 # J/mol * K
        }
    ]
}

# data retrieval function. looks at current temp and species name
    # and returns correct dictionary of coefficients and ref values
    # input: T(K), species_name (string)
    # logic: loops thru list of regimes for that species and finds where 
    # t_min <= T <= t_max
    # output: A dictionary containing the 8 Shomate coeffs, h_form, and s_ref
def get_regime_data(T, species):
    """
    Retrieves the dictionary of coeffs for the valid temp regime.
    Ensures the code respects boundaries.
    """
    if species not in SPECIES_DATA:
        raise ValueError(f"Species '{species}' not found in database.")

    regimes = SPECIES_DATA[species]

    for regime in regimes:
        t_min, t_max = regime['range']
        # Use inclusive bounds for transition points
        if t_min <= T <= t_max:
            return regime

    # Fallback for temperatures slightly oustide NIST range
    if T < 298.15:
        return regimes[0]
    else:
        return regimes[-1]

# Enthalpy. this function implements first half of shomate eqn. 
    # Input: T(K), species_name (string)
    # Logic: Use coeffs A-H in enthalpy formula 
    # output: enthalpy of formation at temp T conv to J/mol
def get_enthalpy(T, species):
    """
    Calculates ΔfH(T) in J/mol using Shomate Eqn.
    This represents the total enthalpy (Formation + Sensible).
    """
    # 1. Get data for current T
    data = get_regime_data(T, species)
    A, B, C, D, E, F, G, H = data['coeffs']

    # 2. Convert T to t (kiloKelvin) for Shomate
    t = T / 1000.0

    # 3. Perform the calculation (Results in kJ/mol)
    # h = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H
    h_kj_mol = (A * t) +  \
               (B * t**2 / 2) +  \
               (C * t**3 / 3) +  \
               (D * t**4 / 4) -  \
               (E / t) + F - H + \
               (data['h_form'] / 1000.0)

    # 4. Convert to J/mol for consistency w global units
    return h_kj_mol * 1000.0

# Entropy. this function uses 2nd half of shomate eqn
    # input: T(K), P(atm), species_name (string)
    # logic: uses coeffs A-H in entropy formula, then apply P correction
    # output: abs entropy at (T,P) in J/mol*K
def get_entropy(T, P, species):
    """
    Calculates the absolute entropy S(T, P) in J/mol*K.
    Inludes the temp-dependent Shomate NIST term and 
    pressure correction term: -R * ln(P/P_ref).
    """
    # 1. Fetch data and coeffs
    data = get_regime_data(T, species)
    A, B, C, D, E, F, G, H = data['coeffs']

    # 2. Temp scaling for Shomate
    t = T / 1000.0

    # 3. Calculate S_temp (etropy at (T, 1atm))
    # S = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2*t^2) + G
    s_temp = (A * np.log(t)) + \
             (B * t) + \
             (C * (t**2) / 2) + \
             (D * (t**3) / 3) - \
             (E / (2 * t**2)) + G

    # 4. Apply pressure correction
    # s_tot = s_temp - R * ln(P / P_ref)
    s_tot = s_temp - R * np.log(P / P_REF)

    return s_tot

# calc potential function
    # input: T(K), P(atm)
    # logic: call H and S functions for each species. finds:
        # ΔH_rxn= N_h2o * h_h2o -(N_h2 * h_h2 + N_o2 * h_o2)
        # ΔS_rxn = N_h2o * s_h2o -(N_h2 * s_h2 + N_o2 * s_o2)
        # ΔG = ΔH - T * ΔS
        # E = -ΔG / (n * F)
    # output: E_rev (Volts)
def calculate_potential(T, P):
    """
    Calculates rev potential E (Volts) for H2 + 0.5 O2 -> H2O(g).
    T: temp in Kelvin
    P: Pressure in atm
    """
    # 1. Get H and S for all species at T and P
    h_h2o = get_enthalpy(T, 'H2O_gas')
    s_h2o = get_entropy(T, P, 'H2O_gas')

    h_h2 = get_enthalpy(T, 'H2')
    s_h2 = get_entropy(T, P, 'H2')

    h_o2 = get_enthalpy(T, 'O2')
    s_o2 = get_entropy(T, P, 'O2')

    # 2. Calculate ΔH_rxn and ΔS_rxn at T
    # = Σ N_p * H_p - Σ N_r * H_r, = Σ N_p * S_p - Σ N_r * S_r
    delta_h = N_H2O * h_h2o - (N_H2 * h_h2 + N_O2 * h_o2)
    delta_s = N_H2O * s_h2o - (N_H2 * s_h2 + N_O2 * s_o2)

    # 3. Calculate ΔG = ΔH - TΔS
    delta_g = delta_h - (T * delta_s)

    # 4. Calculate Rev Potential E = -ΔG / (n * F)
    e_rev = -delta_g / (n * F)

    return e_rev

# Visualization: plot results
    # input: arrays for x and y, and strings for titles
    # output: formatted matplotlib window
def plot_results(x_data, y_data, x_label, y_label, title, filename, is_log=False):
    """
    Standardized plotting function for fuel cell performance graphs.
    """
    plt.figure(figsize=(10, 6))

    if is_log:
        plt.semilogx(x_data, y_data, 'r-o', linewidth=2, markersize=4, label='Reversible Potential')
    else:
        plt.plot(x_data, y_data, 'b-o', linewidth=2, markersize=4, label='Reversible Potential')
    
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()

    # Save the figure for report
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as {filename}")
    plt.show()

# main block:
    # 1 - generate a linspace for T from 298.15 to 1073.15 K,
    # calculate_potential(T, 1.0) for each
    # 2 - call the plotting function to plot this data
    # 3 - create a logspace for P form 0.1 to 100 atm, 
    # call calculate_potential(650, P) 
    # 4 - call plotting function for plot 2
def main():
    # --- Q1: Potential vs. Temp (25ᵒC to 800ᵒC) ---
    # Define T range in Celsius and conver to K
    temps_c = np.linspace(25, 800, 50)
    temps_k = temps_c + CONVERT_C_TO_K

    # Calculate potential at P = 1.0 atm for each T
    e_vs_t = [calculate_potential(T, 1.0) for T in temps_k]
    plot_results(
        temps_c, e_vs_t,
        'Temperature (ᵒC)', 'Reversible Potential (V)',
        'Q1: Reversible Potential vs. Temperature ($P=1$ atm)',
        'Q1_Potential_vs_T.png'
    )

    # --- Q2: Potential vs. Pressure (0.1 to 100 atm) ---
    # Define P range on a log scale for better distr.
    pressures_atm = np.logspace(-1, 2, 50)

    # Calculate potential at T = 650 K for each P
    e_vs_p = [calculate_potential(650, P) for P in pressures_atm]

    plot_results(
        pressures_atm, e_vs_p,
        'Pressure (atm)', 'Reversible Potential (V)',
        'Q2: Reversible Potential vs. Pressure ($T=650$ K)',
        'Q2_Potential_vs_P.png',
        is_log=True
    )


main()

