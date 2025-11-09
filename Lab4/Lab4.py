"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, expon

# -------------------------------
# Data Loading
# -------------------------------
def load_data(file_path):
    """Load CSV dataset from given path and return as a pandas dataframe"""
    return pd.read_csv(file_path)

# -------------------------------
# Descriptive Statistics
# -------------------------------
def calculate_descriptive_stats(data, column): 
    col = data[column].dropna() # This line selects the column and drops missing values
    mean = col.mean()
    median = col.median()
    mode_val = col.mode()[0] # Takes the first mode(max repeating) if multiple  exist
    std = col.std() # Standard deviation
    var = col.var() 
    rng = col.max() - col.min() # To find the range calcs. max-min
    q1 = col.quantile(0.25) # Calcs. First quartile
    q2 = col.quantile(0.5)  # Calcs. second quartile
    q3 = col.quantile(0.75)
    iqr = q3 - q1  # Calcs. interquartile range
    skew = col.skew() #skewness
    kurt = col.kurtosis() # kurtosis

    print(f"=== Descriptive Statistics for {column} ===")
    print(f"mean: {mean:.2f}")
    print(f"median: {median:.2f}")
    print(f"mode: {mode_val:.2f}")
    print(f"std: {std:.2f}")
    print(f"variance: {var:.2f}")
    print(f"range: {rng:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"skewness: {skew:.2f}")
    print(f"kurtosis: {kurt:.2f}")
    print(f"min: {col.min():.2f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q2: {q2:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"max: {col.max():.2f}\n")

    return mean, std, q1, q2, q3  # Returns key statistics for further use

# -------------------------------
# Distribution Plot
# -------------------------------
def plot_distribution(data, column):
    col = data[column].dropna() # Extract the specified column and remove missing values
    mean, median, mode_val = col.mean(), col.median(), col.mode()[0] # We calculate these for plotting reference lines

    plt.figure(figsize=(8,5)) # Creates a new figure with a specific size (width=8, height=5 inch)
    plt.hist(col, bins=30, density=True, color='skyblue', alpha=0.7, edgecolor='black') # Plot a histogram of the data in 'col
    # bins=30 -> divides the data into 30 intervals
    # density=True means: normalize histogram so total area=1 (which is useful for overlaying probability distributions). Barların yüksekliği artık bu aralıkta kaç tane veri olduğunu değil olasılık yoğunluğunu gösteriyor.
    # color bar içi rengi, alpha is for transparency, edgecolor sütunların kenarını siyah yap dedim burda

    plt.axvline(mean, color='red', linestyle='--', label='Mean') # Draws a vertical line at the mean value of the data
    # linestyle='--' makes it dashed, label='Mean' for legend
    plt.axvline(median, color='green', linestyle='-.', label='Median') # Draws a vertical line at the median value
    plt.axvline(mode_val, color='orange', linestyle=':', label='Mode') 
    plt.title(f'{column} Distribution') # Sets the title of the plot using the column name
    plt.xlabel(column) # Labels the x-axis with the column name
    plt.ylabel('Density') # Labels the y-axis as 'Density' since histogram is normalized when we said density=true
    plt.legend()
    plt.savefig('concrete_distribution.png', dpi=300, bbox_inches='tight') # savess the plot as png with high resolution 300 dpi
    plt.show()

# -------------------------------
# Distribution Fitting
# -------------------------------
def plot_distribution_fitting(data, column, fitted_dist):
    col = data[column].dropna()
    plt.figure(figsize=(8,5)) # Creates a figure of size 8x5 inches
    plt.hist(col, bins=30, density=True, color='skyblue', alpha=0.7, edgecolor='black')
    x = np.linspace(col.min(), col.max(), 100) # Generates 100 evenly spaced values from min to max of the data
    plt.plot(x, norm.pdf(x, fitted_dist[0], fitted_dist[1]), 'r--', lw=2, label='Fitted Normal')
    # Overlays the fitted normal distribution
    # fitted_dist[0] = mean of fitted normal, fitted_dist[1] = std deviation
    plt.title(f'{column} Distribution with Fitted Normal')
    plt.xlabel(column)
    plt.ylabel('Density') # Labels y-axis as density because histogram is normalized
    plt.legend()
    plt.savefig('concrete_fitted.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Material Comparison Boxplot
# -------------------------------
def plot_material_comparison(data, column='yield_strength_mpa', group_column='material_type'):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=group_column, y=column, hue=group_column, data=data, palette='Set2', dodge=False, legend=False) # dodge=False keeps each group’s box centered instead of side-by-side
    # legend=False hides repeated legend (unnecessary here)
    # hue gives colours to groups

    plt.title('Material Strength Comparison')
    plt.xlabel(group_column)
    plt.ylabel(column)
    plt.savefig('material_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Probability Distributions Plot
# -------------------------------
def plot_probability_distributions():
    plt.figure(figsize=(10,6))

 # BINOMIAL DISTRIBUTION (Discrete)
    #  Formula for binomial  P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
    x_bin = np.arange(0, 11) # x values for the Binomial PMF (possible outcomes from 0 to 10)
    
    y_bin = binom.pmf(x_bin, 10, 0.3) # Computes the Binomial pmf for n=10 trials, p=0.3 success possibility
    plt.plot(x_bin, y_bin, 'bo-', label='Binomial PMF') # bo- means blue "b" circle "o" and straight line "-"

# POISSON DISTRIBUTION (Discrete)
    # Poisson models the number of events occurring in a fixed interval when events happen independently and at a constant average rate.
    # Formula:  P(X=k) = (e^-λ * λ^k) / k!
    # λ (lambda) = average event rate (mean number of events)
    
    x_pois = np.arange(0, 21) # x values for the Poisson PMF (0 to 20 events)
    y_pois = poisson.pmf(x_pois, 5) # Computes Poisson PMF with λ = 5
    plt.plot(x_pois, y_pois, 'r*-', label='Poisson PMF') # 'r*-' → red stars with a solid connecting linee

# NORMAL DISTRIBUTION (Continuous)
    # Normal models measurement-based variables: strengths, loads, etc.
    # Formula (PDF):  f(x) = 1 / (σ√(2π)) * exp(-(x-μ)^2 / (2σ^2))

    x_norm = np.linspace(0, 10, 100) # Generates 100 smooth points for curve
    plt.plot(x_norm, norm.pdf(x_norm, 5, 1), 'g--', label='Normal PDF') # mean μ = 5, standard deviation σ = 1

# EXPONENTIAL DISTRIBUTION (Continuous)
    # Exponential models time between events, often used for reliability.
    # Formula:  f(x) = λ * e^(-λx)
    # scale parameter = 1/λ, and equals mean of the distribution

    x_exp = np.linspace(0, 10, 100)
    plt.plot(x_exp, expon.pdf(x_exp, scale=2), 'm-.', label='Exponential PDF') # mean = scale = 2 → λ = 1/2
    # 'm-.' → magenta dash-dot line

    plt.title('Probability Distributions')
    plt.xlabel('x')
    plt.ylabel('Probability / Density')
    plt.legend()
    plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Statistical Summary Dashboard
# -------------------------------
def plot_statistical_dashboard(concrete_data):
    col = concrete_data['strength_mpa'].dropna()
    fig, axs = plt.subplots(2,2, figsize=(12,8)) # Creates a 2x2 grid of subplots for the dashboard with a larger figure size
# axs[0,0] = top-left, axs[0,1] = top-right, axs[1,0] = bottom-left, axs[1,1] = bottom-right

# Histogram (Top-Left)
    # bins=20  splits data range into 20 intervals and counts occurrences in bins
    # This gives a quick visual of data distribution: skewness, peaks, spread
    axs[0,0].hist(col, bins=20, color='skyblue', edgecolor='black')
    axs[0,0].set_title('Histogram')
# Boxplot (Top-Right)-
    # Boxplot visualizes the median, quartiles, and potential outliers
    # Useful for comparing spread and detecting extreme values

    axs[0,1].boxplot(col) #matplotlib takes col and calcs Q1,Q2,Q3 
    axs[0,1].set_title('Boxplot')

 # QQ-Plot (Bottom-Left)
    # QQ-Plot checks if the data follows a normal distribution. Points should approximately follow a straight line if data is normal
    # For instance if there is a outlier or right skew it shows us it doent follow str path. right skew yukanı doğru ani sapma
    stats.probplot(col, dist="norm", plot=axs[1,0])
    axs[1,0].set_title('QQ Plot')

# Kernel Density Estimate (KDE) Plot (Bottom-Right)
    # KDE estimates the probability density function of the data
    # Smooth curve helps visualize the shape of the distribution
    sns.kdeplot(col, ax=axs[1,1], color='green')
    axs[1,1].set_title('Density Plot')

    plt.tight_layout() # Adjusts layout to prevent overlapping titles, axes, and labels
    plt.savefig('statistical_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# Probability Calculations
# -------------------------------
def calculate_probability_binomial(n, p, k):
    #   Formula it uses: P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
    # returns us the prob. of k successes in n trials.
    prob = binom.pmf(k, n, p)
    print(f"P(X={k}) [Binomial] = {prob:.4f}")
    return prob

def calculate_probability_poisson(lambda_param, k): # lambda_param = average event rate (λ)
    # k = number of events we want the probability for
    prob = poisson.pmf(k, lambda_param) # poisson.pmf() applies the Poisson formula: P(X=k) = (e^-λ * λ^k) / k!
    print(f"P(X={k}) [Poisson] = {prob:.4f}")
    return prob

def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    # norm.cdf(x, mean, std) gives P(X <= x)

    # Case 1: Two-sided probability → P(x_lower < X < x_upper)
    if x_lower is not None and x_upper is not None:
        prob = norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std) # Subtracts CDFs to get the probability of an interval
        print(f"P({x_lower}<X<{x_upper}) [Normal] = {prob:.4f}")
    # Case 2: One-sided probability → P(X > x_lower)
    elif x_lower is not None:
        prob = 1 - norm.cdf(x_lower, mean, std) # Right-tail probability = 1 - CDF
        print(f"P(X>{x_lower}) [Normal] = {prob:.4f}")
    return prob

def calculate_probability_exponential(mean, x):
    # Exponential distribution formula: f(x) = λ * exp(-λx), x >= 0 ---- Defines how fast events occur on average
    lam = 1/mean # λ (lambda) is the rate parameter, reciprocal of mean. λ (lambda) = 1/mean

    # The cumulative distribution function (CDF) of Exponential: formula  P(X < x) = 1 - e^(-λ * x) ---- It gives the probability that the random variable X happens before x
    prob = 1 - np.exp(-lam*x) # Calculates the cumulative probability P(X < x)
    print(f"P(X<{x}) [Exponential] = {prob:.4f}")
    return prob

def apply_bayes_theorem(prior, sensitivity, specificity):
    # Bayes formula: P(Disease | Positive) = (Sensitivity * Prior) / Denominator
    # Sensitivity = P(Test Positive | Disease) = True Positive Rate
    numerator = sensitivity * prior    # numerator = P(Positive | Disease) * P(Disease)
    denominator = numerator + (1 - specificity)*(1 - prior) # Denominator = P(Test Positive) diseased and positiive+ healthy and positive
    posterior = numerator / denominator
    print(f"Posterior probability [Bayes] = {posterior:.4f}")
    return posterior
"""
def bayes_tree_plot(prior, sensitivity, specificity):
    
    Calculate posterior probability and visualize as a probability tree diagram.
    prior: base rate (P(Damage))
    sensitivity: P(Test+ | Damage)
    specificity: P(Test- | No Damage)

    # Posterior hesapla
    posterior = apply_bayes_theorem(prior, sensitivity, specificity)

    # Tree diagram çizimi
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('off')  # axes gizle

    # Tree textlerini ekle
    ax.text(0.1, 0.9, f'Damage (D) {prior*100:.1f}%', fontsize=12)
    ax.text(0.1, 0.7, f'No Damage (~D) {(1-prior)*100:.1f}%', fontsize=12)

    # Sensitivity & 1-specificity
    ax.text(0.4, 0.95, f'Test+ | D {sensitivity*100:.1f}%', fontsize=12)
    ax.text(0.4, 0.85, f'Test- | D {(1-sensitivity)*100:.1f}%', fontsize=12)
    ax.text(0.4, 0.75, f'Test+ | ~D {(1-specificity)*100:.1f}%', fontsize=12)
    ax.text(0.4, 0.65, f'Test- | ~D {specificity*100:.1f}%', fontsize=12)

    # Posterior sonucu
    ax.text(0.7, 0.9, f'P(D | Test+) = {posterior*100:.2f}%', fontsize=12, color='red')

    plt.title('Bayes Theorem Probability Tree')
    plt.savefig('bayes_tree.png', dpi=300, bbox_inches='tight')
    plt.show()"""

# -------------------------------
# Report File
# -------------------------------
def create_statistical_report(mean,std,q1,q2,q3,file='lab4_statistical_report.txt'):
    with open(file,'w') as f: # Opens a text file in write mode; 'with' ensures the file closes automatically
        f.write("Lab 4 Statistical Report\n")
        f.write("=======================\n")
        f.write(f"Mean: {mean:.2f}\nStd: {std:.2f}\n") # Writes mean and standard dev. with 2 decimals
        f.write(f"Q1: {q1:.2f}, Q2: {q2:.2f}, Q3: {q3:.2f}\n")
    print(f"Report saved to {file}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    concrete_data = load_data(r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\GitHub\CE49X_Fall2025_Koncuk_Berat\Lab4\datasets\concrete_strength.csv")
    materials = load_data(r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\GitHub\CE49X_Fall2025_Koncuk_Berat\Lab4\datasets\material_properties.csv")
    loads = load_data(r"C:\Users\Berat Koncuk\OneDrive - Arup\Masaüstü\GitHub\CE49X_Fall2025_Koncuk_Berat\Lab4\datasets\structural_loads.csv")

    # Concrete Analysis
    mean, std, q1, q2, q3 = calculate_descriptive_stats(concrete_data, 'strength_mpa') # Calculates descriptive statistics for 'strength_mpa' column in concrete_data
    plot_distribution(concrete_data, 'strength_mpa') # Plots the histogram with mean, median, and mode overlay for concrete strength
    plot_distribution_fitting(concrete_data, 'strength_mpa', (mean,std)) # Plots the histogram overlaid with fitted normal distribution using mean and std

    # Material Comparison
    plot_material_comparison(materials) # Creates boxplot comparing yield strengths of different materials

    # Probability Distributions
    plot_probability_distributions()  # Plots  Binomial, Poisson, Normal, Exponential prob. distr.

    # Statistical Dashboard
    plot_statistical_dashboard(concrete_data) # Generates a dashboard with histogram, boxplot, QQ-plot, and density plot for concrete strength

    # Probability Calculations
    print("\n=== Probability Calculations ===")
    calculate_probability_binomial(100, 0.05, 3) # Calculates binomial probability P(X=3) for n=100 trials and p=0.05
    calculate_probability_poisson(10, 8) # Calculate Poisson probability P(X=8) for lambda=10
    calculate_probability_normal(250,15,x_lower=280) # Calculate right-tail probability P(X>280) for normal distribution with mean=250, std=15
    calculate_probability_exponential(1000,500) # Calculate probability P(X<500) for exponential distribution with mean=1000
    apply_bayes_theorem(0.05,0.95,0.90)  # Applies Bayes theo. to get posterior prob. given prior, sensitivity, specificity

    # Create Report
    create_statistical_report(mean,std,q1,q2,q3)

if __name__=="__main__":
    main()
