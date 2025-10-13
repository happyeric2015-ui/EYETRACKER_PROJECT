"""
Eye-Tracking Data Analysis & Dashboard
Processes CSV output from eye_experiment.py and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class EyeTrackingAnalyzer:
    """Analyze and visualize eye-tracking experiment data"""

    def __init__(self, csv_filepath):
        """Initialize analyzer with data file"""
        self.csv_filepath = csv_filepath
        self.raw_data = None
        self.cleaned_data = None
        self.analysis_data = None

    def load_data(self):
        """Load raw data from CSV"""
        try:
            self.raw_data = pd.read_csv(self.csv_filepath)
            print(f"Data loaded successfully from {self.csv_filepath}")
            print(f"Total trials: {len(self.raw_data)}")
            print(f"Columns: {list(self.raw_data.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found: {self.csv_filepath}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_data(self):
        """Remove flagged trials with poor gaze quality"""
        if self.raw_data is None:
            print("Error: No data loaded")
            return False

        # Filter out flagged trials
        self.cleaned_data = self.raw_data[self.raw_data['is_flagged_bad'] == False].copy()

        print(f"\nData Cleaning:")
        print(f"  Original trials: {len(self.raw_data)}")
        print(f"  Flagged bad trials: {len(self.raw_data) - len(self.cleaned_data)}")
        print(f"  Clean trials: {len(self.cleaned_data)}")
        print(f"  Retention rate: {len(self.cleaned_data) / len(self.raw_data) * 100:.1f}%")

        return True

    def simulate_age_groups(self):
        """Simulate Teenager vs Older Adult comparison"""
        if self.cleaned_data is None:
            print("Error: No cleaned data available")
            return False

        # Teenager group (original data)
        teenager_data = self.cleaned_data.copy()
        teenager_data['age_group'] = 'Teenager'

        # Older Adult group (simulated with performance penalty)
        older_data = self.cleaned_data.copy()
        older_data['age_group'] = 'Older Adult'

        # Apply performance penalties to older adults
        # Add 40ms to Antisaccade RT
        antisaccade_mask = older_data['trial_type'] == 'Antisaccade'
        older_data.loc[antisaccade_mask, 'RT_ms'] = older_data.loc[antisaccade_mask, 'RT_ms'] + 40

        # Subtract 10% from Antisaccade accuracy
        # Convert some correct trials to errors
        antisaccade_correct = (older_data['trial_type'] == 'Antisaccade') & (older_data['accuracy'] == 'Correct')
        num_to_flip = int(antisaccade_correct.sum() * 0.10)

        if num_to_flip > 0:
            indices_to_flip = older_data[antisaccade_correct].sample(n=num_to_flip).index
            older_data.loc[indices_to_flip, 'accuracy'] = 'Error'

        # Combine both groups
        self.analysis_data = pd.concat([teenager_data, older_data], ignore_index=True)

        print(f"\nAge Group Simulation:")
        print(f"  Teenager trials: {len(teenager_data)}")
        print(f"  Older Adult trials: {len(older_data)}")
        print(f"  Total analysis trials: {len(self.analysis_data)}")

        return True

    def calculate_summary_stats(self):
        """Calculate summary statistics for each condition"""
        if self.analysis_data is None:
            print("Error: No analysis data available")
            return None

        # Calculate mean RT and accuracy for each combination
        summary = []

        for age_group in ['Teenager', 'Older Adult']:
            for trial_type in ['Prosaccade', 'Antisaccade']:
                subset = self.analysis_data[
                    (self.analysis_data['age_group'] == age_group) &
                    (self.analysis_data['trial_type'] == trial_type)
                ]

                mean_rt = subset['RT_ms'].mean()
                accuracy_pct = (subset['accuracy'] == 'Correct').sum() / len(subset) * 100

                summary.append({
                    'age_group': age_group,
                    'trial_type': trial_type,
                    'mean_RT_ms': mean_rt,
                    'accuracy_pct': accuracy_pct,
                    'n_trials': len(subset)
                })

        summary_df = pd.DataFrame(summary)
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))

        return summary_df

    def create_visualizations(self):
        """Generate all required plots"""
        if self.analysis_data is None:
            print("Error: No analysis data available")
            return False

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 5)

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Mean RT Comparison
        self.plot_mean_rt(axes[0])

        # Plot 2: Mean Accuracy Comparison
        self.plot_mean_accuracy(axes[1])

        # Plot 3: Quality Control Scatter
        self.plot_quality_control(axes[2])

        plt.tight_layout()
        plt.savefig('eye_tracking_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved: eye_tracking_analysis_dashboard.png")

        plt.show()

        return True

    def plot_mean_rt(self, ax):
        """Plot 1: Mean RT Comparison Bar Chart"""
        # Prepare data for grouped bar chart
        summary = []
        for age_group in ['Teenager', 'Older Adult']:
            for trial_type in ['Prosaccade', 'Antisaccade']:
                subset = self.analysis_data[
                    (self.analysis_data['age_group'] == age_group) &
                    (self.analysis_data['trial_type'] == trial_type)
                ]
                mean_rt = subset['RT_ms'].mean()
                se_rt = subset['RT_ms'].sem()

                summary.append({
                    'Age Group': age_group,
                    'Trial Type': trial_type,
                    'Mean RT (ms)': mean_rt,
                    'SE': se_rt
                })

        summary_df = pd.DataFrame(summary)

        # Create grouped bar chart
        x = np.arange(len(['Teenager', 'Older Adult']))
        width = 0.35

        teen_pro = summary_df[(summary_df['Age Group'] == 'Teenager') &
                              (summary_df['Trial Type'] == 'Prosaccade')]['Mean RT (ms)'].values[0]
        teen_anti = summary_df[(summary_df['Age Group'] == 'Teenager') &
                               (summary_df['Trial Type'] == 'Antisaccade')]['Mean RT (ms)'].values[0]
        older_pro = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                               (summary_df['Trial Type'] == 'Prosaccade')]['Mean RT (ms)'].values[0]
        older_anti = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                (summary_df['Trial Type'] == 'Antisaccade')]['Mean RT (ms)'].values[0]

        teen_pro_se = summary_df[(summary_df['Age Group'] == 'Teenager') &
                                 (summary_df['Trial Type'] == 'Prosaccade')]['SE'].values[0]
        teen_anti_se = summary_df[(summary_df['Age Group'] == 'Teenager') &
                                  (summary_df['Trial Type'] == 'Antisaccade')]['SE'].values[0]
        older_pro_se = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                  (summary_df['Trial Type'] == 'Prosaccade')]['SE'].values[0]
        older_anti_se = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                   (summary_df['Trial Type'] == 'Antisaccade')]['SE'].values[0]

        bars1 = ax.bar(x - width/2, [teen_pro, older_pro], width,
                      label='Prosaccade', color='#3498db', alpha=0.8,
                      yerr=[teen_pro_se, older_pro_se], capsize=5)
        bars2 = ax.bar(x + width/2, [teen_anti, older_anti], width,
                      label='Antisaccade', color='#e74c3c', alpha=0.8,
                      yerr=[teen_anti_se, older_anti_se], capsize=5)

        ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Mean RT Comparison by Age Group and Trial Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Teenager', 'Older Adult'])
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=9)

    def plot_mean_accuracy(self, ax):
        """Plot 2: Mean Accuracy Comparison Bar Chart"""
        # Prepare data
        summary = []
        for age_group in ['Teenager', 'Older Adult']:
            for trial_type in ['Prosaccade', 'Antisaccade']:
                subset = self.analysis_data[
                    (self.analysis_data['age_group'] == age_group) &
                    (self.analysis_data['trial_type'] == trial_type)
                ]
                accuracy_pct = (subset['accuracy'] == 'Correct').sum() / len(subset) * 100

                # Calculate standard error for binomial proportion
                p = accuracy_pct / 100
                n = len(subset)
                se = np.sqrt(p * (1 - p) / n) * 100 if n > 0 else 0

                summary.append({
                    'Age Group': age_group,
                    'Trial Type': trial_type,
                    'Accuracy (%)': accuracy_pct,
                    'SE': se
                })

        summary_df = pd.DataFrame(summary)

        # Create grouped bar chart
        x = np.arange(len(['Teenager', 'Older Adult']))
        width = 0.35

        teen_pro = summary_df[(summary_df['Age Group'] == 'Teenager') &
                              (summary_df['Trial Type'] == 'Prosaccade')]['Accuracy (%)'].values[0]
        teen_anti = summary_df[(summary_df['Age Group'] == 'Teenager') &
                               (summary_df['Trial Type'] == 'Antisaccade')]['Accuracy (%)'].values[0]
        older_pro = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                               (summary_df['Trial Type'] == 'Prosaccade')]['Accuracy (%)'].values[0]
        older_anti = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                (summary_df['Trial Type'] == 'Antisaccade')]['Accuracy (%)'].values[0]

        teen_pro_se = summary_df[(summary_df['Age Group'] == 'Teenager') &
                                 (summary_df['Trial Type'] == 'Prosaccade')]['SE'].values[0]
        teen_anti_se = summary_df[(summary_df['Age Group'] == 'Teenager') &
                                  (summary_df['Trial Type'] == 'Antisaccade')]['SE'].values[0]
        older_pro_se = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                  (summary_df['Trial Type'] == 'Prosaccade')]['SE'].values[0]
        older_anti_se = summary_df[(summary_df['Age Group'] == 'Older Adult') &
                                   (summary_df['Trial Type'] == 'Antisaccade')]['SE'].values[0]

        bars1 = ax.bar(x - width/2, [teen_pro, older_pro], width,
                      label='Prosaccade', color='#2ecc71', alpha=0.8,
                      yerr=[teen_pro_se, older_pro_se], capsize=5)
        bars2 = ax.bar(x + width/2, [teen_anti, older_anti], width,
                      label='Antisaccade', color='#f39c12', alpha=0.8,
                      yerr=[teen_anti_se, older_anti_se], capsize=5)

        ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Mean Accuracy Comparison by Age Group and Trial Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Teenager', 'Older Adult'])
        ax.set_ylim([0, 105])
        ax.legend(frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    def plot_quality_control(self, ax):
        """Plot 3: Quality Control Scatter Plot (RT vs Saccade Velocity)"""
        if self.raw_data is None:
            print("Error: No raw data available for quality control plot")
            return

        # Use raw data to show flagged vs clean trials
        clean_trials = self.raw_data[self.raw_data['is_flagged_bad'] == False]
        flagged_trials = self.raw_data[self.raw_data['is_flagged_bad'] == True]

        # Plot clean trials
        ax.scatter(clean_trials['RT_ms'], clean_trials['saccade_velocity'],
                  alpha=0.6, s=50, c='#3498db', label='Clean Trials', edgecolors='black', linewidth=0.5)

        # Plot flagged trials
        ax.scatter(flagged_trials['RT_ms'], flagged_trials['saccade_velocity'],
                  alpha=0.8, s=80, c='#e74c3c', label='Flagged (Bad Quality)',
                  marker='X', edgecolors='darkred', linewidth=1)

        ax.set_xlabel('Reaction Time (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Saccade Velocity (units/s)', fontsize=12, fontweight='bold')
        ax.set_title('Quality Control: RT vs Saccade Velocity', fontsize=14, fontweight='bold')
        ax.legend(frameon=True, shadow=True, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add text annotation
        ax.text(0.05, 0.95, f'Flagged: {len(flagged_trials)} trials\nClean: {len(clean_trials)} trials',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 60)
        print("Eye-Tracking Data Analysis Dashboard")
        print("=" * 60)

        # Step 1: Load data
        if not self.load_data():
            return False

        # Step 2: Clean data
        if not self.clean_data():
            return False

        # Step 3: Simulate age groups
        if not self.simulate_age_groups():
            return False

        # Step 4: Calculate summary statistics
        summary_stats = self.calculate_summary_stats()

        # Step 5: Create visualizations
        if not self.create_visualizations():
            return False

        print("\n" + "=" * 60)
        print("Analysis complete!")
        print("=" * 60)

        return True


def main():
    """Main entry point"""
    # Default CSV filename
    csv_file = "eye_tracking_results_DATA.csv"

    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: Data file not found: {csv_file}")
        print("Please run eye_experiment.py first to generate data.")
        return

    # Create analyzer and run analysis
    analyzer = EyeTrackingAnalyzer(csv_file)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
