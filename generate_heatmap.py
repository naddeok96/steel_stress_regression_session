import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the Excel file
file_path = "data/leitner_model_16_hidden_layers_kfold_loss_0_0008_stainless_steel_304_exhaustive_predictions_10.xlsx"
df = pd.read_excel(file_path)

# Extract the columns
temp = df["Temp"]
strain = df["Strain"]
strain_rate = df["Strain Rate"]
stress = df["Predictions"]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot with a specific color range
sc = ax.scatter(temp, strain, strain_rate, c=stress, cmap="viridis", vmin=min(stress), vmax=max(stress))

# Set labels for each axis
ax.set_xlabel('Temp')
ax.set_ylabel('Strain')
ax.set_zlabel('Strain Rate')

# Add color bar legend
cbar = plt.colorbar(sc, orientation="vertical")
cbar.set_label('Stress')

# Show the plot
plt.savefig(file_path.replace('.xlsx', '.png'))
