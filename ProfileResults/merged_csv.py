import pandas as pd
import json

# Load the new JSON files
file_paths = [
    "clFFTResultOnNvidiaH100.json",
    "cuFFTResultOnNvidiaH100.json",
    "occaResultOnNvidiaH100.json"
]

# Load data from the JSON files
results = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        results.append(data)

# Convert the loaded data into DataFrames
clfft_results = results[0]
cufft_results = results[1]
occa_results = results[2]

clfft_df = pd.DataFrame(list(clfft_results.items()), columns=["Experiment", "clFFT (ns)"])
cufft_df = pd.DataFrame(list(cufft_results.items()), columns=["Experiment", "cuFFT (ns)"])
occa_df = pd.DataFrame(list(occa_results.items()), columns=["Experiment", "OCCA (ns)"])

# Function to extract window size, platform, and batch size from the experiment key
def extract_info(experiment, platform_type):
    if platform_type == 'CLFFT' or platform_type == 'CUFFT':
        window_size = int(experiment[:experiment.index('C')])  # Capture the window size as an integer
        platform = experiment[experiment.index('C'):(experiment.index('C')+5)]  # Platform name (CLFFT, CUFFT)
        batch_size = int(experiment[experiment.index('C')+5:])  # The remaining part is the batch size as an integer
    else:  # For occa
        window_size = int(experiment[:experiment.index('o')])  # Capture the window size as an integer
        platform = 'occa'
        batch_size = int(experiment[experiment.index('o')+4:])  # The remaining part is the batch size as an integer
    return window_size, platform, batch_size

# Apply the function to extract the relevant fields in each DataFrame
clfft_df[['WindowSize', 'Platform', 'BatchSize']] = clfft_df['Experiment'].apply(lambda x: pd.Series(extract_info(x, 'CLFFT')))
cufft_df[['WindowSize', 'Platform', 'BatchSize']] = cufft_df['Experiment'].apply(lambda x: pd.Series(extract_info(x, 'CUFFT')))
occa_df[['WindowSize', 'Platform', 'BatchSize']] = occa_df['Experiment'].apply(lambda x: pd.Series(extract_info(x, 'occa')))

# Drop the 'Platform' column since it's redundant (all 'CLFFT' for clfft_df, etc.)
clfft_df.drop(columns=['Platform'], inplace=True)
cufft_df.drop(columns=['Platform'], inplace=True)
occa_df.drop(columns=['Platform'], inplace=True)

# Merge the dataframes on 'WindowSize' and 'BatchSize'
merged_df = pd.merge(clfft_df, cufft_df, on=['WindowSize', 'BatchSize'], how='inner')
merged_df = pd.merge(merged_df, occa_df, on=['WindowSize', 'BatchSize'], how='inner')

# Drop the redundant 'Experiment' columns
merged_df.drop(columns=['Experiment_x', 'Experiment_y', 'Experiment'], inplace=True)

# Reorder the columns as needed: WindowSize, BatchSize, clFFT, cuFFT, OCCA
merged_df = merged_df[['WindowSize', 'BatchSize', 'clFFT (ns)', 'cuFFT (ns)', 'OCCA (ns)']]

# Sort the dataframe by 'WindowSize' and 'BatchSize' in ascending order
merged_df = merged_df.sort_values(by=['WindowSize', 'BatchSize'], ascending=[True, True])

# Save to CSV
merged_df.to_csv('merged_fft_results.csv', index=False)

# Print success message
print("Data merged, sorted by WindowSize and BatchSize in ascending order, and saved as 'merged_fft_results.csv'.")
