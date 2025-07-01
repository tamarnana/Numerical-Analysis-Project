import matplotlib.pyplot as plt

def plot_defect_trend(defects):
    """
    Plot the trend of defects over time.

    Parameters:
    defects (list or array-like): Sequence of defect counts indexed by time.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(defects, marker='o', color='red')
    plt.title("Defect Trend Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Number of Defects")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_code_churn_vs_defects(code_churn, defects):
    """
    Scatter plot showing relationship between code churn and defects.

    Parameters:
    code_churn (list or array-like): Values representing amount of code churn.
    defects (list or array-like): Corresponding defect counts.
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(code_churn, defects, c='blue')
    plt.title("Code Churn vs. Defects")
    plt.xlabel("Code Churn")
    plt.ylabel("Defects")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_test_coverage_vs_defects(test_coverage, defects):
    """
    Plot test coverage percentage against defects.

    Parameters:
    test_coverage (list or array-like): Test coverage percentages.
    defects (list or array-like): Corresponding defect counts.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(test_coverage, defects, marker='x', linestyle='--', color='green')
    plt.title("Test Coverage vs. Defects")
    plt.xlabel("Test Coverage (%)")
    plt.ylabel("Defects")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
