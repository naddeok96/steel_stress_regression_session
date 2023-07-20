import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def open_data(filename):
    # Open data and return it 
    data = pd.read_excel(filename)
    return data

def clean_data(data):
    # check for any values that are not numbers
    tensor_data = torch.tensor(data.values).float()  # convert DataFrame to PyTorch Tensor
    nan_locations = torch.isnan(tensor_data)
    if nan_locations.any():
        print("NaN values found at:")
        rows, cols = torch.where(nan_locations)
        for row, col in zip(rows, cols):
            print(f"Row {row.item()}, Column {data.columns[col.item()]}")
    tensor_data[nan_locations] = 0  # replace NaNs with 0
    return tensor_data

def calculate_standardization_values(data, filename, headers):
    # calculate mean and std of each col and save in an xlsx and print save name and return values
    mean_values = torch.mean(data, dim=0)
    std_values = torch.std(data, dim=0)
    mean_std_values = torch.stack([mean_values, std_values])
    df = pd.DataFrame(mean_std_values.numpy(), columns=headers, index=['mean', 'std'])
    df.to_excel(filename.replace('.xlsx', '_standardization_values.xlsx'))  # save with filename reference
    print(f"Standardization values saved as {filename.replace('.xlsx', '_standardization_values.xlsx')}")
    return mean_values, std_values

def standardize_data(data, mean_values, std_values):
    # standardize each column by subtracting mean and dividing by std
    standardized_data = (data - mean_values) / std_values
    return standardized_data

def save_data(data, filename, headers):
    # save as xlsx in same folder
    pd.DataFrame(data.numpy(), columns=headers).to_excel(filename)
            
def analyze_input_space(data, output_file_prefix, headers):
    # Function to perform k-means clustering for each column/variable and save the results

    def calculate_cluster_info(clusters, data):
        cluster_info = []
        for cluster_id in range(clusters):
            cluster_data = data[cluster_labels == cluster_id]
            cluster_mean = torch.mean(cluster_data, dim=0)
            cluster_std = torch.std(cluster_data, dim=0)
            num_data_points = cluster_data.shape[0]
            cluster_info.append((num_data_points, cluster_mean.numpy()[0], cluster_std.numpy()[0]))
        return cluster_info

    num_variables = data.shape[1] - 1  # Last column is the target variable

    # Create an empty DataFrame to hold the cluster information
    df_clusters = pd.DataFrame(columns=['Variable', 'Number of Data Points', 'Mean', 'Std'],
                               dtype='object')

    with pd.ExcelWriter(output_file_prefix) as writer:
        for col_idx in range(num_variables):
            variable_data = data[:, col_idx]
            X = variable_data.unsqueeze(1)  # Convert to 2D tensor for clustering

            # Determine the number of clusters using silhouette score
            best_score, best_clusters = -1, 0
            max_clusters = min(10, X.shape[0] - 1)  # Maximum number of clusters is n-1
            for clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_clusters = clusters

            # Perform k-means clustering with the best number of clusters
            kmeans = KMeans(n_clusters=best_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)

            # Save cluster information to the DataFrame
            cluster_info = calculate_cluster_info(best_clusters, X)

            # Sort the cluster_info based on the mean value
            cluster_info.sort(key=lambda x: x[1])

            for info in cluster_info:
                df_clusters = df_clusters.append({'Variable': headers[col_idx],
                                                  'Number of Data Points': info[0],
                                                  'Mean': info[1],
                                                  'Std': info[2]}, ignore_index=True)

            # Add a blank row between unique variables
            if col_idx < num_variables - 1:
                df_clusters = df_clusters.append({col: '' for col in df_clusters.columns}, ignore_index=True)

        # Save the cluster information to a sheet named "Cluster Information"
        df_clusters.to_excel(writer, sheet_name='Cluster Information', index=False)
        
def process_data(filename, standardization_values=None):
    filename = filename.replace('\\', '/')
    
    df = open_data(filename)
    tensor_data = clean_data(df)
    save_data(tensor_data, filename.replace('.xlsx', '_cleaned.xlsx'), df.columns)
    
    # Perform k-means clustering and save the results
    cluster_output_file = filename.replace('.xlsx', '_clusters.xlsx')
    analyze_input_space(tensor_data, cluster_output_file, df.columns)

    if standardization_values is None:
        mean_values, std_values = calculate_standardization_values(tensor_data, filename, df.columns)
    else:
        mean_values, std_values = standardization_values

    standardized_data = standardize_data(tensor_data, mean_values, std_values)
    save_data(standardized_data, filename.replace('.xlsx', '_standardized.xlsx'), df.columns)

    return standardized_data

if "__main__" == __name__:
    process_data("data\stainless_steel_304.xlsx", standardization_values=None)

