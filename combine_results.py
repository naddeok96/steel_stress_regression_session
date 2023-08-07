import pandas as pd
import os
import numpy as np
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
    elif "exponential_gpr_linear" in name:
        return "Exponential GPR (Linear)"
    elif "exponential_gpr_constant" in name:
        return "Exponential GPR (Constant)"
    elif "linear_svm" in name:
        return "Linear SVM"
    else:
        return name.replace('_', ' ').title()

def format_latex(df):
    # Get the minimum values for each column
    min_values = df.min(numeric_only=True)

    # Get the maximum values for each column, excluding infinities
    max_values = df[df < np.inf].max(numeric_only=True)

    # Get the minimum value in the "Test Loss (MSE)" column, excluding zeros
    min_test_loss = df[df['Test Loss (MSE)'] > 0]['Test Loss (MSE)'].min()

    # Start building the LaTeX table
    latex_str = '\\begin{table}[h]\n\\centering\n\\begin{tabularx}{\\columnwidth}{lXXXXX}\n\\toprule\n'
    headers = [
        "Model",
        "\\rotatebox{315}{\\makebox[0pt][r]{Average K-Fold MSE}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Standard Deviation of K-Fold MSE}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Test Loss (MSE)}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Ratio (Avg K-Fold MSE / Test Loss)}}",
        "\\rotatebox{315}{\\makebox[0pt][r]{Training Time (Seconds)}}",
    ]
    latex_str += ' & '.join(headers) + ' \\\\\n\\midrule\n'

    # Iterate through the rows and format each cell
    for index, row in df.iterrows():
        model_name = get_model_name(index)
        row_str = model_name
        for col, value in row.items():
            if value > 1 and value != np.inf:
                cell_str = f'{value:.5g}'
            else:
                cell_str = f'{value:.5f}'

            if value == min_values[col] and value != 0:
                cell_str = '\\cellcolor{green!25}\\textbf{' + cell_str + '}'
            if value == max_values[col] and value < np.inf:
                cell_str = '\\cellcolor{red!25}' + cell_str
            if col == 'Test Loss (MSE)' and value == 0.0:
                cell_str = '\\cellcolor{yellow!25}' + cell_str
            if col == 'Ratio (Avg K-Fold MSE / Test Loss)' and value == np.inf:
                cell_str = '\\cellcolor{yellow!25}' + cell_str
            if col == 'Test Loss (MSE)' and value == min_test_loss:
                cell_str = '\\cellcolor{green!25}\\textbf{' + cell_str + '}'
            row_str += ' & ' + cell_str
        latex_str += row_str + ' \\\\\n'

    latex_str += '\\bottomrule\n\\end{tabularx}\n\\caption{Comparison of various regression models on the given dataset.}\n\\label{tab:comparison}\n\\end{table}'
    return latex_str

# Initialize a list to store individual DataFrames
dfs = []

# List of filenames to be skipped/ignored
skip_files = ["models/model_16_hidden_layers_100_kfold_loss_0_031753.pt", "data/leitner_model_16_hidden_layers_100_epochs_kfold_loss_0_014002_results.xlsx"]

# Iterate through all files in the "data/" directory
for filename in os.listdir('data/'):
    if filename in skip_files:
        continue
    if filename.endswith("_results.xlsx"):
        file_path = os.path.join('data/', filename)
        df = pd.read_excel(file_path)
        df.index = [filename] # Index is set to filename, will be converted later
        df['Ratio (Avg K-Fold MSE / Test Loss)'] = df['Average K-Fold MSE'] / df['Test Loss (MSE)']
        dfs.append(df)

# Concatenate all the DataFrames in the list
combined_results = pd.concat(dfs)

# Reorder the columns to make "Training Time (Seconds)" the last column
cols = [col for col in combined_results.columns if col != "Training Time (Seconds)"] + ["Training Time (Seconds)"]
combined_results = combined_results[cols]

# Format the DataFrame for LaTeX
latex_str = format_latex(combined_results)

# Save the LaTeX string to a file
with open('data/combined_results.tex', 'w') as f:
    f.write(latex_str)

# Save the combined results to a new Excel file
combined_results.to_excel("data/combined_results.xlsx", index=True, float_format="%.7f")
