import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def detect_outliers_zscore(data, threshold=3):
    z_scores = stats.zscore(data)
    outliers = data[abs(z_scores) > threshold]
    return outliers

def detect_outliers_tukeys_fence(data):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]
    return outliers

def mad(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad

def detect_outliers_mad(data, threshold=3.5):
    median = np.median(data)
    mad_value = mad(data)
    mad_scores = 1.4826 * np.abs(data - median) / mad_value
    outliers = data[mad_scores > threshold]
    return outliers

# Replace 'directory_path' with the path to the directory containing the files.
directory_path = 'data'

# Find all files ending with 'individual_losses.xlsx'
files = glob.glob(os.path.join(directory_path, '*individual_losses.xlsx'))

# Threshold values to vary
threshold_values = np.linspace(2, 6, 9)

# Colors and line styles for each file name
colors = plt.cm.tab20.colors[:len(files)]
linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.']

# Initialize a list to store the results for all models and threshold values
results = {file: [] for file in files}

# Loop through each file and perform outlier detection with varying thresholds
for file in files:
    df = pd.read_excel(file)  # Assuming there's only one column in the Excel file
    column_name = df.columns[0]
    data = df[column_name]

    for threshold in threshold_values:
        # Perform outlier detection using Z-Score method
        outliers_zscore = detect_outliers_zscore(data, threshold)
        results[file].append((threshold, len(outliers_zscore)))

        # Perform outlier detection using Tukey's Fence method
        outliers_tukeys_fence = detect_outliers_tukeys_fence(data)
        results[file].append((threshold, len(outliers_tukeys_fence)))

        # Perform outlier detection using MAD (Median Absolute Deviation) method
        outliers_mad = detect_outliers_mad(data, threshold)
        results[file].append((threshold, len(outliers_mad)))

# Plot the results in three separate plots and the blank plot for the legend
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for idx, (file, data) in enumerate(results.items()):
    thresholds, num_outliers = zip(*data)
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    axs[0].plot(thresholds, num_outliers, color=color, linestyle=linestyle, marker='o', label=os.path.basename(file).replace('_individual_losses.xlsx', ''))

axs[0].set_title('Z-Score Outliers')
axs[0].set_xlabel('Threshold')
axs[0].set_ylabel('Number of Outliers')
axs[0].grid(True)

for idx, (file, data) in enumerate(results.items()):
    thresholds, num_outliers = zip(*data)
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    axs[1].plot(thresholds, num_outliers, color=color, linestyle=linestyle, marker='o', label=os.path.basename(file).replace('_individual_losses.xlsx', ''))

axs[1].set_title('Tukey\'s Fence Outliers')
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Number of Outliers')
axs[1].grid(True)

for idx, (file, data) in enumerate(results.items()):
    thresholds, num_outliers = zip(*data)
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    axs[2].plot(thresholds, num_outliers, color=color, linestyle=linestyle, marker='o', label=os.path.basename(file).replace('_individual_losses.xlsx', ''))

axs[2].set_title('MAD Outliers')
axs[2].set_xlabel('Threshold')
axs[2].set_ylabel('Number of Outliers')
axs[2].grid(True)

# Create a blank plot for the legend
axs[3].axis('off')
axs[3].legend(*axs[0].get_legend_handles_labels(), loc='center')
axs[3].set_title('Legend')

plt.tight_layout()
plt.savefig('data/outliers_vs_thresholds.png')
plt.show()
