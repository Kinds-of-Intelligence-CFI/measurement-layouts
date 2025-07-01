import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_and_analyze_data(folder_path="."):
    """
    Load all CSV files matching the pattern and analyze agent performance
    """
    # Pattern to match your files (accounting for decimal values in noise)
    pattern = os.path.join(folder_path, "results_pixels_*_noise_*.*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None, None
    
    print(f"Found {len(files)} files")
    
    # Load all data
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("No data could be loaded")
        return None, None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Define pass condition (finalReward > -0.99)
    combined_df['passed'] = combined_df['finalReward'] > -0.99
    
    # Calculate pass rates by pixels
    pixels_summary = combined_df.groupby('pixelInput').agg({
        'passed': ['sum', 'count']
    }).round(4)
    pixels_summary.columns = ['passes', 'total_tests']
    pixels_summary['pass_rate'] = pixels_summary['passes'] / pixels_summary['total_tests']
    pixels_summary = pixels_summary.reset_index()
    
    # Calculate pass rates by noise
    # noise_summary = combined_df[combined_df['pixelInput'] == 40].groupby('navigationNoise').agg({
    noise_summary = combined_df.groupby('navigationNoise').agg({
        'passed': ['sum', 'count']
    }).round(4)
    noise_summary.columns = ['passes', 'total_tests']
    noise_summary['pass_rate'] = noise_summary['passes'] / noise_summary['total_tests']
    noise_summary = noise_summary.reset_index()
    
    return pixels_summary, noise_summary

def create_subplot_charts(pixels_data, noise_data, save_path=None):
    """
    Create subplot with two bar charts
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Pass rate by pixels
    bars1 = ax1.bar(pixels_data['pixelInput'], pixels_data['pass_rate'], 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Pixel Input', fontsize=12)
    ax1.set_ylabel('Proportion of Tests Passed', fontsize=12)
    ax1.set_title('Agent Performance by Pixel Input', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, pixels_data['pass_rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Set x-tick labels to show all pixel values
    ax1.set_xticks(pixels_data['pixelInput'])
    ax1.set_xticklabels(pixels_data['pixelInput'])
    
    # Plot 2: Pass rate by noise
    # Create x positions for bars (use indices instead of actual noise values)
    x_positions = range(len(noise_data))
    bars2 = ax2.bar(x_positions, noise_data['pass_rate'], 
                    color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Navigation Noise', fontsize=12)
    ax2.set_ylabel('Proportion of Tests Passed', fontsize=12)
    ax2.set_title('Agent Performance by Navigation Noise', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars2, noise_data['pass_rate'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Set x-tick labels to show all noise values
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'{x:.1f}' for x in noise_data['navigationNoise']])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    plt.show()
    
    return fig

def print_summary_stats(pixels_data, noise_data):
    """
    Print summary statistics
    """
    print("\n=== SUMMARY STATISTICS ===")
    print("\nPass Rates by Pixel Input:")
    print(pixels_data.to_string(index=False))
    
    print("\nPass Rates by Navigation Noise:")
    print(noise_data.to_string(index=False))
    
    print(f"\nOverall Statistics:")
    print(f"Best performing pixel input: {pixels_data.loc[pixels_data['pass_rate'].idxmax(), 'pixelInput']} "
          f"({pixels_data['pass_rate'].max():.3f} pass rate)")
    print(f"Worst performing pixel input: {pixels_data.loc[pixels_data['pass_rate'].idxmin(), 'pixelInput']} "
          f"({pixels_data['pass_rate'].min():.3f} pass rate)")
    
    print(f"Best performing noise level: {noise_data.loc[noise_data['pass_rate'].idxmax(), 'navigationNoise']:.1f} "
          f"({noise_data['pass_rate'].max():.3f} pass rate)")
    print(f"Worst performing noise level: {noise_data.loc[noise_data['pass_rate'].idxmin(), 'navigationNoise']:.1f} "
          f"({noise_data['pass_rate'].min():.3f} pass rate)")

# Main execution
if __name__ == "__main__":
    # Set the folder path where your CSV files are located
    folder_path = "./data/"  # Change this to your folder path if different
    
    # Load and analyze data
    pixels_data, noise_data = load_and_analyze_data(folder_path)
    
    if pixels_data is not None and noise_data is not None:
        # Print summary statistics
        print_summary_stats(pixels_data, noise_data)
        
        # Create and display the charts
        fig = create_subplot_charts(pixels_data, noise_data, 
                                  save_path="agent_performance_analysis.png")
        
        print("\nAnalysis complete!")
    else:
        print("Could not analyze data. Please check that your CSV files are in the correct location.")