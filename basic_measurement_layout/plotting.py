import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns # Import seaborn for better heatmap visualization

def load_and_prepare_data(folder_path="."):
    """
    Load all CSV files matching the pattern and return a combined DataFrame.
    """
    # Pattern to match your files (accounting for decimal values in noise)
    # The pattern is adjusted to correctly capture the noise value, e.g., "noise_0.1.csv"
    pattern = os.path.join(folder_path, "results_pixels_*_noise_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(files)} files")
    
    # Load all data
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Extract pixelInput and navigationNoise from the filename as a fallback
            # This logic assumes filename format: results_pixels_X_noise_Y.Z.csv
            if 'pixelInput' not in df.columns or 'navigationNoise' not in df.columns:
                parts = os.path.basename(file).replace('.csv', '').split('_')
                if len(parts) >= 5: # Ensure enough parts to extract info
                    try:
                        df['pixelInput'] = int(parts[2])
                        # Handle noise which might be like '0.1', '1.0', etc.
                        noise_str = parts[4]
                        if len(parts) > 5: # If there's a decimal part like noise_0.1
                            noise_str += '.' + parts[5]
                        df['navigationNoise'] = float(noise_str)
                    except ValueError as ve:
                        print(f"Could not parse pixelInput or navigationNoise from {file}: {ve}")
                        continue # Skip this file if parsing fails
                else:
                    print(f"Filename format unexpected for {file}. Skipping.")
                    continue
            all_data.append(df)
        except Exception as e:
            print(f"Error reading or processing {file}: {e}")
    
    if not all_data:
        print("No data could be loaded")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Define pass condition (finalReward > -0.99) for potential future use
    combined_df['passed'] = combined_df['finalReward'] > -0.99
    
    return combined_df

def create_performance_grid(df, save_path=None):
    """
    Create a grid of histograms showing the distribution of finalReward 
    for each combination of pixelInput and navigationNoise.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot create performance grid plot.")
        return None

    # Get unique sorted values for pixels and noise
    pixel_levels = sorted(df['pixelInput'].unique())
    noise_levels = sorted(df['navigationNoise'].unique())
    
    num_pixels = len(pixel_levels)
    num_noise = len(noise_levels)
    
    if num_pixels == 0 or num_noise == 0:
        print("Not enough data diversity to create a grid.")
        return None

    # Create a grid of subplots
    fig, axes = plt.subplots(num_pixels, num_noise, 
                             figsize=(num_noise * 4, num_pixels * 3), 
                             sharex=True, sharey=True)
    
    # Flatten axes array for easier iteration if it's 1D
    if num_pixels == 1 and num_noise == 1:
        axes = np.array([[axes]]) # Make it 2D even for a single subplot
    elif num_pixels == 1:
        axes = np.atleast_2d(axes)
    elif num_noise == 1:
        axes = np.atleast_2d(axes).T # Transpose to get correct shape for single column

    fig.suptitle('Agent Performance: Distribution of Final Rewards', fontsize=20, fontweight='bold')

    for i, pixel in enumerate(pixel_levels):
        for j, noise in enumerate(noise_levels):
            ax = axes[i, j]
            
            # Filter data for the specific pixel and noise combination
            subset = df[(df['pixelInput'] == pixel) & (df['navigationNoise'] == noise)]
            
            if not subset.empty:
                # Create a histogram of finalReward
                ax.hist(subset['finalReward'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                
                # Add a vertical line for the mean reward
                mean_reward = subset['finalReward'].mean()
                ax.axvline(mean_reward, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_reward:.2f}')
                ax.legend(fontsize='small')
            else:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, fontsize=12, color='gray')

            # Set titles for each subplot
            ax.set_title(f'Pixels: {pixel}, Noise: {noise:.1f}', fontsize=10)
            
            # Set y-axis label only for the first column
            if j == 0:
                ax.set_ylabel('Frequency', fontsize=10)
            
            # Set x-axis label only for the last row
            if i == num_pixels - 1:
                ax.set_xlabel('Final Reward', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for subtitle
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
        
    plt.show()
    
    return fig

def create_performance_heatmap(df, save_path=None):
    """
    Create a heatmap showing the average finalReward for each combination 
    of pixelInput and navigationNoise.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot create heatmap.")
        return None

    # Calculate the average finalReward for each combination of pixelInput and navigationNoise
    average_performance = df.groupby(['pixelInput', 'navigationNoise'])['finalReward'].mean().reset_index()

    # Pivot the table to prepare data for the heatmap
    # pixelInput will be rows (index), navigationNoise will be columns, values will be average finalReward
    heatmap_data = average_performance.pivot(index='pixelInput', columns='navigationNoise', values='finalReward')

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Average Final Reward'})
    
    plt.title('Average Agent Performance Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Navigation Noise', fontsize=12)
    plt.ylabel('Pixel Input', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
    plt.yticks(rotation=0) # Keep y-axis labels horizontal
    
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        
    plt.show()
    
    return plt.gcf() # Return the current figure

# Main execution
if __name__ == "__main__":
    # Set the folder path where your CSV files are located
    folder_path = "./data/"  # Change this to your folder path if different
    
    # Load and prepare data
    combined_data = load_and_prepare_data(folder_path)
    
    if combined_data is not None:
        # Create and display the grid chart
        print("\nGenerating performance grid chart...")
        fig_grid = create_performance_grid(combined_data, 
                                           save_path="agent_performance_grid.png")
        
        # Create and display the heatmap
        print("\nGenerating performance heatmap...")
        fig_heatmap = create_performance_heatmap(combined_data,
                                                 save_path="agent_performance_heatmap.png")
        
        if fig_grid and fig_heatmap:
            print("\nAnalysis complete! Both charts generated successfully.")
        else:
            print("Failed to generate one or both plots.")
    else:
        print("\nCould not analyze data. Please check that your CSV files are in the correct location and format.")