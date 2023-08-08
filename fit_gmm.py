import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set the evaluation timeout to 10 seconds (or any other suitable value)
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10000"

# Read the data from the Excel file
file_path = 'data/stainless_steel_304_cleaned.xlsx'
data = pd.read_excel(file_path)

# Define a range of number of components to try
n_components_range = range(1, 6) # You can adjust this range

def plot_gmm_1d(gmm, X, col_name, criterion_name):
    x_grid = np.linspace(X.min(), X.max(), 1000)
    x_buffer = 0.1 * (X.max() - X.min())
    x_grid = np.linspace(X.min() - x_buffer, X.max() + x_buffer, 1000)

    means = gmm.means_.flatten()
    variances = gmm.covariances_.flatten()

    pdfs = []
    for w, mean, var in zip(gmm.weights_, means, variances):
        if var <= 1e-6:
            pdf = np.zeros_like(x_grid)
            pdf[x_grid == mean] = np.inf
        else:
            pdf = w * norm.pdf(x_grid, mean, np.sqrt(var))
        pdfs.append(pdf)

    peak_cap = 1.5 * np.max([np.nanmax(pdf) for pdf in pdfs if np.isfinite(np.nanmax(pdf))])
    total_pdf = np.sum(pdfs, axis=0)
    total_pdf[total_pdf > peak_cap] = peak_cap

    for i, pdf in enumerate(pdfs):
        pdf[pdf > peak_cap] = peak_cap
        plt.plot(x_grid, pdf)

    plt.plot(x_grid, total_pdf, color='black', linestyle='--')

    vertical_offset = 0.1 * peak_cap
    texts = []
    arrows = []
    for i, (w, mean, var) in enumerate(zip(gmm.weights_, means, variances)):
        peak_value = w * norm.pdf(mean, mean, np.sqrt(var)) if var > 1e-6 else peak_cap
        peak_value = min(peak_value, peak_cap)
        text_content = f'$\\mu_{{{i}}}: {mean:.3f}$\n$\\sigma^2_{{{i}}}: {var:.3f}$\n$\\pi_{{{i}}}: {w:.3f}$'
        text = plt.text(mean, peak_value + vertical_offset, text_content, fontsize=8, horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.8))
        texts.append(text)

        # Create an arrow object and append it to the arrows list
        arrow = plt.annotate("", xy=(mean, peak_value), xytext=(mean, peak_value + vertical_offset),
                             arrowprops=dict(arrowstyle='->', color='black'))
        arrows.append(arrow)

    # Use the adjust_text library to avoid overlaps
    adjust_text(texts, add_objects=arrows)

    plt.title(f'{col_name} ({criterion_name})')
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.ylim(0, peak_cap + 2 * vertical_offset)
    plt.xlim(X.min() - x_buffer, X.max() + x_buffer)  # Set the x-limits with the buffer
    plt.tight_layout()
    plt.savefig(f"data/gmm_{col_name}_{criterion_name}.png")
    plt.close()

def generate_random_angles(num_angles):
    angles = []
    for _ in range(num_angles):
        azimuth = np.random.uniform(0, 360)
        elevation = np.random.uniform(0, 90)
        angles.append((azimuth, elevation))
    return angles

def plot_gmm_3d(gmm, feature_names, criterion_name, num_points=1000, num_angles=9):
    fig = plt.figure(figsize=(15, 15))
    
    # Calculate points for each component outside the loop
    points = np.zeros((num_points * gmm.n_components, 3))
    for n in range(gmm.n_components):
        mean = gmm.means_[n]
        cov = gmm.covariances_[n]
        rv = multivariate_normal(mean, cov)
        
        # Sample points from the current Gaussian component
        points[n * num_points: (n + 1) * num_points] = rv.rvs(num_points)

    angles = generate_random_angles(num_angles)

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(3, 3, i + 1, projection='3d')
        for n in range(gmm.n_components):
            # Scatter points from each Gaussian component in the same subplot
            start_idx = n * num_points
            end_idx = (n + 1) * num_points
            ax.scatter(points[start_idx:end_idx, 0], points[start_idx:end_idx, 1], points[start_idx:end_idx, 2], s=5, label=f'Component {n+1}')

        method_name = 'Random'
        view_number = i + 1
        ax.set_title(f"{feature_names[0]}, {feature_names[1]}, and {feature_names[2]} - {method_name} - View {view_number}")
        ax.set_xlabel(f'{method_name} 1')
        ax.set_ylabel(f'{method_name} 2')
        ax.set_zlabel(f'{method_name} 3')

        # Set the viewpoint for each subplot
        ax.view_init(*angle)
        ax.legend()

    plt.suptitle(f"{feature_names[0]}, {feature_names[1]}, and {feature_names[2]} - {criterion_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"data/gmm_collective_{criterion_name}_views.png")


# Define a function to save results
def save_results(gmm, X, col_name, criterion_name, sheet_writer):
    results = {
        'Column': [col_name],
        'Criterion': [criterion_name],
        'Number of Gaussians': [gmm.n_components],
        'AIC': [gmm.aic(X)],
        'BIC': [gmm.bic(X)],
        'Means': [str(gmm.means_)],
        'Covariances': [str(gmm.covariances_)],
        'Mixing Coefficients': [str(gmm.weights_)]
    }
    pd.DataFrame.from_dict(results).to_excel(sheet_writer, index=False, sheet_name=f"{col_name}_{criterion_name}")

# Function to fit GMM and calculate the required information
def fit_gmm(X, col_name, sheet_writer):
    best_aic = best_bic = np.inf
    best_gmm_aic = best_gmm_bic = None

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(X)
        aic = gmm.aic(X)
        bic = gmm.bic(X)

        if aic < best_aic:
            best_aic = aic
            best_gmm_aic = gmm

        if bic < best_bic:
            best_bic = bic
            best_gmm_bic = gmm

    save_results(best_gmm_aic, X, col_name, 'AIC', sheet_writer)
    save_results(best_gmm_bic, X, col_name, 'BIC', sheet_writer)

    # Plotting for individual columns
    if X.shape[1] == 1:
        plot_gmm_1d(best_gmm_aic, X.values, col_name, 'AIC')
        plot_gmm_1d(best_gmm_bic, X.values, col_name, 'BIC')
    # Plotting for collective results
    else:
        feature_names = ["Temp", "Plastic Strain", "Strain Rate"]
        plot_gmm_3d(best_gmm_aic, feature_names, 'AIC')
        plot_gmm_3d(best_gmm_bic, feature_names, 'BIC')

# Open a writer to save to Excel
output_path = 'data/gmm_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # Fit GMM to each column individually (excluding the last column)
    for col in data.columns[:-1]:
        fit_gmm(data[[col]], col, writer)
        print(f"Processed {col}")

    # Fit GMM to all columns collectively (excluding the last column)
    fit_gmm(data.iloc[:, :-1], 'collective', writer)
    print(f"Processed all columns collectively")

print(f"Results saved to {output_path}")



