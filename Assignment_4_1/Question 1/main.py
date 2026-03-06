from data_init import load_and_clean_data, T
from fitting import smooth_data, find_best_fit
from solver import solve_kinetics
from plotting import plot_tafel_analysis

def main() -> None:
    """Main execution pipeline for Tafel plot analysis."""
    
    # Configuration
    file_path = 'ME5895_Asgn4_1_q1data.xlsx'
    
    # 1. Initialize and clean data
    j_raw, v_raw = load_and_clean_data(file_path)
    
    # 2. Smooth data and find regression parameters
    log_j_int, j_smooth, v_smooth = smooth_data(j_raw, v_raw)
    best_N, m_best, b_best, r_sq = find_best_fit(log_j_int, v_smooth)
    
    # 3. Solve for physical constants
    i_0, alpha_c = solve_kinetics(m_best, b_best, T)
    
    # Output to terminal
    print("-" * 40)
    print("KINETICS RESULTS:")
    print(f"Exchange Current Density (I_0): {i_0:.4e} A/m²")
    print(f"Cathodic Transfer Coefficient (alpha_c): {alpha_c:.4f}")
    print("-" * 40)
    
    # 4. Plot final analysis
    plot_tafel_analysis(
        j_raw, v_raw, j_smooth, v_smooth, 
        best_N, m_best, b_best, r_sq, i_0, alpha_c
    )

if __name__ == "__main__":
    main()