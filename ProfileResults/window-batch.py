import pandas as pd
import matplotlib.pyplot as plt

# Load merged CSV file
merged_df = pd.read_csv('merged_fft_results.csv')

# Get the unique window sizes available in the dataset
available_window_sizes = merged_df['WindowSize'].unique()

# Display available window sizes to the user
print("Available Window Sizes:", available_window_sizes)

# 원하는 윈도우 사이즈 입력 받기
window_size_input = int(input("Enter the Window Size you want to plot from the list above: "))

# Filter the dataframe by the given window size
df_window = merged_df[merged_df['WindowSize'] == window_size_input]

# Get the unique batch sizes for the given window size
batch_sizes = df_window['BatchSize'].unique()

# Set up the figure for a large plot for the input window size
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the batch size against the three FFT platforms without connecting lines (scatter plot)
ax.scatter(df_window['BatchSize'], df_window['clFFT (ns)'], label='clFFT (ns)', marker='o')
ax.scatter(df_window['BatchSize'], df_window['cuFFT (ns)'], label='cuFFT (ns)', marker='s')
ax.scatter(df_window['BatchSize'], df_window['OCCA (ns)'], label='OCCA (ns)', marker='^')

# Set titles and labels
ax.set_title(f'Window Size: {window_size_input}')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Execution Time (ns)')
ax.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
