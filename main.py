from input_handler import load_data
from analysis import (
    solve_equation_model,
    find_optimal_testing,
    interpolate_defects,
    integrate_defects
)
from visualization import (
    plot_defect_trend,
    plot_code_churn_vs_defects,
    plot_test_coverage_vs_defects
)

def main():
    filepath = "data/dev_metrics.csv"
    data = load_data(filepath)

    code_churn = data['code_churn']
    test_coverage = data['test_coverage']
    defects = data['defects']

    print("🔍 Solving Defect Prediction Model...")
    beta1, beta2 = solve_equation_model(code_churn, test_coverage, defects)
    print(f"Model Coefficients: β1 = {beta1:.4f}, β2 = {beta2:.4f}")

    print("📉 Finding Optimal Testing Coverage...")
    optimal = find_optimal_testing(test_coverage, defects)
    print(f"Optimal Testing Coverage: {optimal:.2f}")

    print("📈 Interpolating Defect Rate at Midpoint...")
    interp = interpolate_defects(defects)
    print(f"Interpolated defects at midpoint: {interp:.2f}")

    print("📊 Integrating total defects over time...")
    total = integrate_defects(defects)
    print(f"Total integrated defects: {total:.2f}")

    # Visuals
    plot_defect_trend(defects)
    plot_code_churn_vs_defects(code_churn, defects)
    plot_test_coverage_vs_defects(test_coverage, defects)

if __name__ == "__main__":
    main()
