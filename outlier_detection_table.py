import os
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

def get_model_name(filename):
    name = filename.split('_results.xlsx')[0]
    if "linear_regression" in name:
        return "Linear Regression"
    elif "boosted_decision_tree" in name:
        return "Boosted Decision Tree"
    elif "rational_quadratic_gpr" in name:
        return "Rational Quadratic GPR"
    elif "leitner_model_" in name and "epochs" in name:
        epochs = re.search(r'(\d+)_epochs', name)
        if epochs:
            return f"Leitner MLP ({epochs.group(1)} Epochs)"
    elif "model_" in name and "epochs" in name and "leitner_model_" not in name:
        epochs = re.search(r'(\d+)_epochs', name)
        if epochs:
            return f"MLP ({epochs.group(1)} Epochs)"
    elif "square_exponential_gpr" in name:
        return "Square Exponential GPR"
    elif "robust_linear_regression" in name:
        return "Robust Linear Regression"
    elif "quad_svm" in name:
        return "Quad SVM"
    elif "fine_regression_tree" in name:
        return "Fine Regression Tree"
    elif "fine_gaussian_svm" in name:
        return "Fine Gaussian SVM"
    elif "cubic_svm" in name:
        return "Cubic SVM"
    elif "quadratic_svm" in name:
        return "Quadratic SVM"
    elif "gpr_w_linear" in name:
        return "Exponential GPR (Linear)"
    elif "gpr_w_constant" in name:
        return "Exponential GPR (Constant)"
    elif "linear_svm" in name:
        return "Linear SVM"
    else:
        return name.replace('_', ' ').title()
    
def format_latex(df):
    # Compute the minimum and maximum value for each numeric column
    numeric_df = df.drop(columns='File Name').apply(pd.to_numeric, errors='coerce')
    numeric_df = numeric_df.replace(0, pd.NA)
    min_values = numeric_df.min()
    max_values = numeric_df.max()

    latex_str = '\\begin{table}[h]\n\\centering\n\\begin{tabularx}{\\columnwidth}{lXXXXX}\n\\toprule\n'
    headers = [
        "Model",
        "\\rotatebox{315}{\\makebox[0pt][r]{Z-Score Outliers}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Z-Score Outliers Contribution (\\%)}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Tukey's Fence Outliers}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Tukey's Fence Outliers Contribution (\\%)}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Loss Difference}}",
    ]

    latex_str += ' & '.join(headers) + ' \\\\\n\\midrule\n'
    for index, row in df.iterrows():
        file_name = get_model_name(row['File Name'])
        row_str = file_name
        for col, value in row.items():
            if col != 'File Name':
                cell_value = pd.to_numeric(value, errors='coerce')
                cell_str = f'{cell_value:.4g}' if pd.notna(cell_value) else str(value)  # Format with 4 significant figures or keep as is if NaN

                # If the value is numeric and equal to the minimum for the column, apply green cell color and bold format
                if pd.notna(cell_value) and cell_value == min_values[col]:
                    cell_str = f'\\cellcolor{{green!25}}\\textbf{{{cell_str}}}'

                # If the value is numeric and equal to the maximum for the column, apply red cell color
                if pd.notna(cell_value) and cell_value == max_values[col]:
                    cell_str = f'\\cellcolor{{red!25}}{cell_str}'

                # If the value is zero, apply light yellow background
                if cell_value == 0:
                    cell_str = f'\\cellcolor{{yellow!25}}{cell_str}'

                cell_str = cell_str.replace('%', '\\%')  # Escaping percentage symbol in cell values
                row_str += ' & ' + cell_str
        latex_str += row_str + ' \\\\\n'

    latex_str += '\\bottomrule\n\\end{tabularx}\n\\caption{Outliers Summary Table.}\n\\label{tab:outliers_summary}\n\\end{table}'
    return latex_str

def detect_outliers_zscore(data, threshold=3):
    z_scores = stats.zscore(data)
    outliers = data[abs(z_scores) > threshold]
    return outliers

def detect_outliers_tukeys_fence(data, threshold=1.5):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - threshold * iqr
    upper_fence = q3 + threshold * iqr
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

# Initialize a list to store the results for all files
results = []

# Loop through each file and perform outlier detection
for file in files:
    if "model_16" not in file:
        continue
    df = pd.read_excel(file)  # Assuming there's only one column in the Excel file
    column_name = df.columns[0]
    data = df[column_name]

    # Perform outlier detection using Z-Score method
    outliers_zscore = detect_outliers_zscore(data)

    # Perform outlier detection using Tukey's Fence method
    outliers_tukeys_fence = detect_outliers_tukeys_fence(data)

    # Perform outlier detection using MAD (Median Absolute Deviation) method
    outliers_mad = detect_outliers_mad(data)

    # Get the file name without any directories and remove the '_individual_losses.xlsx' part
    file_name = os.path.basename(file).replace('_individual_losses.xlsx', '')

    # Calculate the difference between the maximum and minimum loss
    loss_diff = data.max() - data.min()

    # Calculate the total loss (sum of all data points)
    total_loss = data.sum()

    # Calculate the sum of losses for outliers for each method
    outliers_loss_zscore = outliers_zscore.sum()
    outliers_loss_tukeys = outliers_tukeys_fence.sum()
    outliers_loss_mad = outliers_mad.sum()

    # Calculate the percentage contribution of outliers to the total loss for each method
    if total_loss != 0:
        outliers_contribution_zscore = (outliers_loss_zscore / total_loss) * 100
        outliers_contribution_tukeys = (outliers_loss_tukeys / total_loss) * 100
        outliers_contribution_mad = (outliers_loss_mad / total_loss) * 100
    else:
        # Set contribution to "N/A" if total_loss is zero
        outliers_contribution_zscore = 'N/A'
        outliers_contribution_tukeys = 'N/A'
        outliers_contribution_mad = 'N/A'

    # Save the results to the results list
    result = {
        'File Name': file_name,
        'Z-Score Outliers': len(outliers_zscore),
        'Z-Score Outliers Contribution (%)': outliers_contribution_zscore,
        'Tukey\'s Fence Outliers': len(outliers_tukeys_fence),
        'Tukey\'s Fence Outliers Contribution (%)': outliers_contribution_tukeys,
        # 'MAD Outliers': len(outliers_mad),
        # 'MAD Outliers Contribution (%)': outliers_contribution_mad,
        'Loss Difference': loss_diff
    }
    results.append(result)

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the results to a new Excel file
output_file_name = 'data/outliers_summary_nn_only.xlsx'
results_df.to_excel(output_file_name, index=False, engine='openpyxl')

latex_table_file_name = 'data/outliers_summary_nn_only.tex'
with open(latex_table_file_name, 'w') as f:
    latex_str = format_latex(results_df)
    f.write(latex_str)

print("Outlier detection and saving completed.")