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
    'legend.fontsize': 16, # Legend (smaller for more entries)
    'figure.titlesize': 20 # Figure suptitle
})


def create_brier_score_plot(save_path=None, annotate=False):
    # Load the CSV data into pandas DataFrames
    df_agents = pd.read_csv("AgentResultsAblated.csv")
    df_models = pd.read_csv("logistic_sem_briers.csv")
    
    # Merge the dataframes on the agent name
    df = pd.merge(df_agents, df_models, 
                  left_on='Agent Name', right_on='Agent Name', 
                  how='inner')

    # Extract data for the Success panel
    x_success = df['Average Success']
    y_model_success = df['Model Brier Score Success']
    y_agg_success = df['Aggregate Brier Score Success']
    y_logistic_success = df['Logistic Brier Success']
    y_sem_success = df['SEM (Both Dependents) Brier Success']
    labels = df['Name Short_x']  # Use the one from AgentResultsAblated

    # Extract data for the Choice panel
    x_choice = df['Average Correct Choice']
    y_model_choice = df['Model Brier Score Choice']
    y_agg_choice = df['Aggregate Brier Score Choice']
    y_logistic_choice = df['Logistic Brier Choice']
    y_sem_choice = df['SEM (Both Dependents) Brier Choice']

    # Prepare theoretical Brier score trend line
    x_trend = np.linspace(0, 1, 500)
    y_trend = x_trend * (1 - x_trend)

    # Create subplots: one for Success, one for Choice
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Stylistic parameters for label boxes
    bbox_props = dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8) if annotate else None

    # -------- Success panel --------
    ax = axes[0]
    ax.scatter(x_success, y_model_success, label='Model', color='tab:blue', zorder=2, s=80)
    ax.scatter(x_success, y_agg_success, label='Aggregate', color='tab:orange', zorder=2, s=80)
    ax.scatter(x_success, y_logistic_success, label='Logistic', color='tab:green', zorder=2, s=80)
    ax.scatter(x_success, y_sem_success, label='SEM', color='purple', zorder=2, s=80)
    ax.plot(x_trend, y_trend, color='orange', linestyle='--', label='_nolegend_', zorder=1)
    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Brier Score')
    ax.set_title('Brier Score vs Success Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    ax.legend(loc='lower right')
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
    ax.scatter(x_choice, y_model_choice, label='Model', color='tab:blue', zorder=2, s=80)
    ax.scatter(x_choice, y_agg_choice, label='Aggregate', color='tab:orange', zorder=2, s=80)
    ax.scatter(x_choice, y_logistic_choice, label='Logistic', color='tab:green', zorder=2, s=80)
    ax.scatter(x_choice, y_sem_choice, label='SEM', color='purple', zorder=2, s=80)
    ax.plot(x_trend, y_trend, color='orange', linestyle='--', label='_nolegend_', zorder=1)
    ax.set_xlabel('Choice Accuracy')
    ax.set_title('Brier Score vs Choice Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    ax.legend(loc='lower right')
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
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return df


def print_brier_rankings(df):
    """Print rankings for each agent showing which method performs best (lower is better)"""
    
    print("\n" + "="*80)
    print("BRIER SCORE RANKINGS BY AGENT (Lower is Better)")
    print("="*80)
    
    for idx, row in df.iterrows():
        agent_name = row['Name Short_x']
        
        # Success rankings
        print(f"\n{agent_name} - SUCCESS:")
        success_scores = {
            'Aggregate': row['Aggregate Brier Score Success'],
            'Model': row['Model Brier Score Success'],
            'SEM': row['SEM (Both Dependents) Brier Success'],
            'Logistic': row['Logistic Brier Success']
        }
        success_sorted = sorted(success_scores.items(), key=lambda x: x[1])
        success_ranking = " > ".join([f"{name} ({score:.4f})" for name, score in success_sorted])
        print(f"  {success_ranking}")
        
        # Choice rankings
        print(f"{agent_name} - CHOICE:")
        choice_scores = {
            'Aggregate': row['Aggregate Brier Score Choice'],
            'Model': row['Model Brier Score Choice'],
            'SEM': row['SEM (Both Dependents) Brier Choice'],
            'Logistic': row['Logistic Brier Choice']
        }
        choice_sorted = sorted(choice_scores.items(), key=lambda x: x[1])
        choice_ranking = " > ".join([f"{name} ({score:.4f})" for name, score in choice_sorted])
        print(f"  {choice_ranking}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    # Count wins for success
    success_wins = {'Aggregate': 0, 'Model': 0, 'SEM': 0, 'Logistic': 0}
    for _, row in df.iterrows():
        scores = {
            'Aggregate': row['Aggregate Brier Score Success'],
            'Model': row['Model Brier Score Success'],
            'SEM': row['SEM (Both Dependents) Brier Success'],
            'Logistic': row['Logistic Brier Success']
        }
        winner = min(scores.items(), key=lambda x: x[1])[0]
        success_wins[winner] += 1
    
    print("\nSuccess - Number of times each method had the lowest Brier score:")
    for method, count in sorted(success_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count}")
    
    # Count wins for choice
    choice_wins = {'Aggregate': 0, 'Model': 0, 'SEM': 0, 'Logistic': 0}
    for _, row in df.iterrows():
        scores = {
            'Aggregate': row['Aggregate Brier Score Choice'],
            'Model': row['Model Brier Score Choice'],
            'SEM': row['SEM (Both Dependents) Brier Choice'],
            'Logistic': row['Logistic Brier Choice']
        }
        winner = min(scores.items(), key=lambda x: x[1])[0]
        choice_wins[winner] += 1
    
    print("\nChoice - Number of times each method had the lowest Brier score:")
    for method, count in sorted(choice_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count}")
    
    print("\n" + "="*80 + "\n")

# Main execution
if __name__ == "__main__":
    # Set the folder path where your CSV files are located
    folder_path = "./data/"  # Change this to your folder path if different

    df = create_brier_score_plot(save_path="figures/op_agents_brier_score_plot_extended.png", annotate=False)
    print_brier_rankings(df)