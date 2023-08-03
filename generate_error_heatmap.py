import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the Excel file
file_path = "data/leitner_model_16_hidden_layers_kfold_loss_0_0008_stainless_steel_304.xlsx"
df = pd.read_excel(file_path)

# Extract the columns
temp = df["Temp"]
strain = df["Plastic Strain"]
strain_rate = df["Strain Rate"]
error = df["Absolute Error per Row"]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D scatter plot with a specific color range
sc = ax.scatter(temp, strain, strain_rate, c=error, cmap="cool", vmin=min(error), vmax=max(error))

# Set labels for each axis
ax.set_xlabel('Temp')
ax.set_ylabel('Strain')
ax.set_zlabel('Strain Rate')

# Create a new axes for the color bar
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7]) # You can adjust the values to position the color bar

# Add color bar legend
cbar = plt.colorbar(sc, cax=cbar_ax, orientation="vertical")
cbar.set_label('Error')

# Show the plot
# plt.show()
plt.savefig(file_path.replace('.xlsx', '_error.png'))
