import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optionally import adjustText to avoid label overlap
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

plt.rcParams.update({
    'font.size': 20,       # Default text size
    'axes.titlesize': 20,  # Subplot title
    'axes.labelsize': 20,  # X and Y labels
    'xtick.labelsize': 20, # X tick labels
    'ytick.labelsize': 20, # Y tick labels
    'legend.fontsize': 20, # Legend
    'figure.titlesize': 20 # Figure suptitle
})


def create_brier_score_plot(save_path=None, annotate=False):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv("AgentResultsAblated.csv")

    # Extract data for the Success panel
    x_success = df['Average Success']
    y_model_success = df['Model Brier Score Success']
    y_agg_success = df['Aggregate Brier Score Success']
    labels = df['Name Short']

    # Extract data for the Choice panel
    x_choice = df['Average Correct Choice']
    y_model_choice = df['Model Brier Score Choice']
    y_agg_choice = df['Aggregate Brier Score Choice']

    # Prepare theoretical Brier score trend line
    x_trend = np.linspace(0, 1, 500)
    y_trend = x_trend * (1 - x_trend)

    # Create subplots: one for Success, one for Choice
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Stylistic parameters for label boxes
    bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8) if annotate else None

    # -------- Success panel --------
    ax = axes[0]
    ax.scatter(x_success, y_model_success, label='Model', color='tab:blue', zorder=2)
    ax.scatter(x_success, y_agg_success, label='Aggregate', color='tab:orange', zorder=2)
    ax.plot(x_trend, y_trend, color='orange', linestyle='--', label='_nolegend_', zorder=1)
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score vs Success Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    ax.legend()
    ax.grid(False)

    # Add labels only for model points with rounded bounding boxes
    if annotate:
        texts = []
        for xi, yi, label in zip(x_success, y_model_success, labels):
            texts.append(ax.text(
                xi, yi, label,
                fontsize=8, ha='right', va='bottom', zorder=3,
                bbox=bbox_props
            ))
        if adjust_text:
            adjust_text(texts, 
                        ax=ax, 
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, shrinkA=5),
                        autoalign='y',
                        force_text=(0, 0.5),
                        expand_points=(1.5,1.5),
                        expand_text=(1.05, 1.05),
                        )
        
    # -------- Choice panel --------
    ax = axes[1]
    ax.scatter(x_choice, y_model_choice, label='Model', color='tab:blue', zorder=2)
    ax.scatter(x_choice, y_agg_choice, label='Aggregate', color='tab:orange', zorder=2)
    ax.plot(x_trend, y_trend, color='orange', linestyle='--', label='_nolegend_', zorder=1)
    ax.set_xlabel('Choice Accuracy')
    ax.set_title('Brier Score vs Choice Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    ax.legend()
    ax.grid(False)

    # Add labels only for model points with rounded bounding boxes
    if annotate:
        texts = []
        for xi, yi, label in zip(x_choice, y_model_choice, labels):
            texts.append(ax.text(
                xi, yi, label,
                fontsize=8, ha='right', va='bottom', zorder=3,
                bbox=bbox_props
            ))
        if adjust_text:
            adjust_text(texts, 
                        ax=ax, 
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, shrinkA=5),
                        autoalign='y',
                        force_text=(0, 0.5),
                        expand_points=(1.5,1.5),
                        expand_text=(1.05, 1.05),
                        )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


# Main execution
if __name__ == "__main__":
    # Set the folder path where your CSV files are located
    folder_path = "./data/"  # Change this to your folder path if different

    create_brier_score_plot(save_path="figures/op_agents_brier_score_plot.png")
