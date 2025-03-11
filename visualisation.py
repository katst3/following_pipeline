import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import warnings
from debugging_functions import *

warnings.simplefilter(action='ignore', category=FutureWarning)

# initialize global variables for model and dataset names
MODEL_NAME = "LSTM"
DATASET_NAME = "all_2dir"

DENSITY_CATEGORIES = ['simple', 'deeper', 'less dense', 'dense', 'superdense']
DENSITY_COLORS = ["#FFC084", "#FC8469", "#D841A8", "#782FB0", "#00108C"]

DATASET_SPLITS = ['Train', 'Valid A', 'Valid Comp', 'Test']
DATASET_COLORS = ["#FFC084", "#D841A8", "#00108C", "#782FB0"]

# utility Functions
def get_categories(data_info):
    """Extract unique categories from data_info dataframes"""
    all_categories = set()
    
    for df_key in ['train_df', 'val_df', 'valb_df', 'test_df']:
        if df_key in data_info and hasattr(data_info[df_key], 'columns'):
            df = data_info[df_key]
            if 'category' in df.columns:
                categories = df['category'].unique()
                # replace 'less' with 'less dense' 
                categories = ['less dense' if cat == 'less' else cat for cat in categories]
                all_categories.update(categories)
    
    if not all_categories:
        return DENSITY_CATEGORIES
    
    ordered_categories = []
    for cat in DENSITY_CATEGORIES:
        if cat in all_categories:
            ordered_categories.append(cat)
            all_categories.discard(cat)
    
    ordered_categories.extend(sorted(all_categories))
    
    return ordered_categories

def get_actor_column(df):
    """Determine which column contains actor count information"""
    for col in ['num_actors', 'n_actors', 'num_nouns']:
        if col in df.columns:
            return col
    return None

def extract_detailed_metrics(eval_results):
    """
    Extract detailed metrics from evaluation results
    Returns a dictionary of metrics by category and actor count
    """
    detailed_metrics = {}
    
    if 'detailed_metrics' in eval_results:
        for category, noun_counts in eval_results['detailed_metrics'].items():
            # replace 'less' with 'less dense' in category names
            category_name = 'less dense' if category == 'less' else category
            detailed_metrics[category_name] = {}
            
            for noun_count_str, accuracy in noun_counts.items():
                try:
                    if isinstance(noun_count_str, str) and noun_count_str.startswith('noun_'):
                        # Format: 'noun_2' -> extract 2
                        noun_count = int(noun_count_str.split('_')[1])
                    else:
                        noun_count = int(noun_count_str)
                    
                    if isinstance(accuracy, (int, float)):
                        detailed_metrics[category_name][noun_count] = accuracy
                    elif isinstance(accuracy, tuple) and len(accuracy) == 2:
                        _, val_acc = accuracy
                        detailed_metrics[category_name][noun_count] = val_acc
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse noun count '{noun_count_str}' - {e}")
    
    return detailed_metrics

def extract_accuracy_data(data_info, eval_results):
    """Extract accuracy data directly from comprehensive results structure"""
    print("\n==== DEBUGGING EXTRACT_ACCURACY_DATA ====")
    
    train_accuracies = eval_results.get('train_accuracies', {})
    validation_accuracies = eval_results.get('validation_accuracies', {})
    validb_accuracies = eval_results.get('validb_accuracies', {})
    test_accuracies = eval_results.get('test_accuracies', {})
    
    # rename 'less' category to 'less dense'
    for acc_dict in [train_accuracies, validation_accuracies, validb_accuracies, test_accuracies]:
        if 'less' in acc_dict:
            acc_dict['less dense'] = acc_dict.pop('less')
    
    print("Extracted accuracy dictionaries:")
    print(f"  Train: {list(train_accuracies.keys()) if train_accuracies else 'Empty'}")
    print(f"  Valid A: {list(validation_accuracies.keys()) if validation_accuracies else 'Empty'}")
    print(f"  Valid Comp: {list(validb_accuracies.keys()) if validb_accuracies else 'Empty'}")
    print(f"  Test: {list(test_accuracies.keys()) if test_accuracies else 'Empty'}")
    
    for name, acc_dict in [
        ("Train", train_accuracies),
        ("Valid A", validation_accuracies),
        ("Valid Comp", validb_accuracies),
        ("Test", test_accuracies)
    ]:
        if acc_dict:
            for category, counts in acc_dict.items():
                sorted_counts = sorted(counts.items())
                if sorted_counts:
                    print(f"  {name}/{category}: {sorted_counts[:3]}...")
    
    print("=========================================\n")
    
    return train_accuracies, validation_accuracies, test_accuracies, validb_accuracies


def calculate_binomial_ci(successes, trials, confidence=0.95):
    """Calculate binomial confidence intervals using scipy."""
    if trials <= 0:
        return 0.5, 0.0, 1.0  # Default for no trials
    
    successes = max(0, min(successes, trials))
    
    try:
        res = scipy.stats.binomtest(successes, trials)
        ci_low, ci_high = res.proportion_ci(confidence)
        proportion = successes / trials
        return proportion, ci_low, ci_high
    except:
        try:
            from scipy.stats import proportion_confint
            proportion = successes / trials
            ci_low, ci_high = proportion_confint(successes, trials, alpha=1-confidence)
            return proportion, ci_low, ci_high
        except:
            proportion = successes / trials
            z = 1.96  # For 95% confidence
            interval = z * np.sqrt((proportion * (1 - proportion)) / trials)
            return proportion, max(0, proportion - interval), min(1, proportion + interval)

def extract_sample_sizes(data_info):
    """
    Extract sample sizes from data_info dataframes with the same structure as accuracy data
    Returns a dictionary of sample sizes by dataset, category, and actor count
    """
    sample_sizes = {
        'Train': {},
        'Valid A': {},
        'Valid Comp': {},
        'Test': {}
    }
    
    dataset_mapping = {
        'train_df': 'Train',
        'val_df': 'Valid A',
        'valb_df': 'Valid Comp',
        'test_df': 'Test'
    }
    
    for df_key, dataset_name in dataset_mapping.items():
        if df_key in data_info and isinstance(data_info[df_key], pd.DataFrame):
            df = data_info[df_key]
            
            actor_column = get_actor_column(df)
            
            if actor_column and 'category' in df.columns:
                for category in df['category'].unique():
                    # Convert 'less' to 'less dense'
                    category_name = 'less dense' if category == 'less' else category
                    category_df = df[df['category'] == category]
                    
                    if category_name not in sample_sizes[dataset_name]:
                        sample_sizes[dataset_name][category_name] = {}
                    
                    try:
                        actor_values = pd.to_numeric(category_df[actor_column], errors='coerce')
                        for actor_count in sorted(actor_values.dropna().unique()):
                            count = len(category_df[category_df[actor_column] == actor_count])
                            sample_sizes[dataset_name][category_name][int(actor_count)] = count
                    except Exception as e:
                        print(f"Error processing actor counts for {category_name}: {str(e)}")
    
    return sample_sizes

def prepare_plot_data(train_accuracies, validation_accuracies, test_accuracies, validb_accuracies, sample_sizes):
    """Aggregate accuracy data with confidence intervals."""
    aggregated_data = {}
    
    def apply_binomial_ci(acc_dict, sample_sizes_dict, set_name):
        data_points = 0
        
        for category, accuracies in acc_dict.items():
            if category not in sample_sizes_dict:
                continue
                
            category_samples = sample_sizes_dict[category]
            
            for noun_count, accuracy in accuracies.items():
                if noun_count not in category_samples:
                    continue
                    
                n_trials = category_samples[noun_count]
                if n_trials <= 0:
                    continue  
                
                n_successes = max(0, min(int(accuracy * n_trials), n_trials))
                
                try:
                    mean, ci_low, ci_high = calculate_binomial_ci(n_successes, n_trials)
                    
                    key = (int(noun_count), set_name, category)
                    if key not in aggregated_data:
                        aggregated_data[key] = {
                            'Accuracy': mean,
                            'CI Lower': ci_low,
                            'CI Upper': ci_high,
                            'Sample Size': n_trials
                        }
                        data_points += 1
                except Exception as e:
                    print(f"Error calculating CI for {category}/{noun_count}: {e}")
    
    if train_accuracies and 'Train' in sample_sizes:
        apply_binomial_ci(train_accuracies, sample_sizes['Train'], 'Train')
    
    if validation_accuracies and 'Valid A' in sample_sizes:
        apply_binomial_ci(validation_accuracies, sample_sizes['Valid A'], 'Valid A')
    
    if validb_accuracies and 'Valid Comp' in sample_sizes:
        apply_binomial_ci(validb_accuracies, sample_sizes['Valid Comp'], 'Valid Comp')
    
    if test_accuracies and 'Test' in sample_sizes:
        apply_binomial_ci(test_accuracies, sample_sizes['Test'], 'Test')
    
    plot_data = []
    for (x_val, set_name, category), data in aggregated_data.items():
        plot_data.append({
            'n_nouns': x_val,
            'Mean': data['Accuracy'],
            'CI Lower': data['CI Lower'],
            'CI Upper': data['CI Upper'],
            'Sample Size': data['Sample Size'],
            'Set': set_name,
            'Category': category
        })
    
    if not plot_data:
        print("WARNING: No plot data was generated. Creating minimal placeholder data.")
    
    df = pd.DataFrame(plot_data)
    return df

def get_line_width(sample_size, max_sample_size):
    """Determine line width based on sample size."""
    if max_sample_size <= 0:
        return 0.4  
    return max(0.4, (sample_size / max_sample_size) * 3)  

def plot_dataset_summary(datasets, dataset_colors, average_accuracy, min_ci, max_ci):
    """Plot summary statistics for each dataset with improved styling."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    for i, dataset in enumerate(datasets):
        plt.fill_betweenx([min_ci[i], max_ci[i]], i - 0.3, i + 0.3, 
                        color=dataset_colors[i % len(dataset_colors)], alpha=0.3)
        
        plt.hlines(average_accuracy[i], i - 0.3, i + 0.3, 
                  colors=dataset_colors[i % len(dataset_colors)], linewidth=2)
        
        plt.text(i, average_accuracy[i] + 0.03, f"{average_accuracy[i]:.4f}", 
                 horizontalalignment='center', 
                 color=dataset_colors[i % len(dataset_colors)], 
                 fontweight='bold',
                 fontsize=12)
    
    plt.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1.3)
    
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('#dddddd')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_color('#dddddd')
    ax.spines['right'].set_linewidth(0.5)
    
    plt.xticks(range(len(datasets)), datasets)
    plt.ylim(0, 1.0)
    plt.xlabel('Dataset', fontsize=15, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=15, fontweight='bold')
    plt.title(f'{MODEL_NAME} - {DATASET_NAME}: Average accuracy across datasets', 
              fontsize=16, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7, color='#dddddd')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_NAME.lower()}_{DATASET_NAME}_dataset_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_category_breakdown(df_plot, categories, category_colors, selected_dataset='Valid Comp'):
    """Plot accuracy breakdown by category for a specific dataset."""
    plt.figure(figsize=(17, 8))
    ax = plt.gca()
    
    dataset_data = df_plot[df_plot['Set'] == selected_dataset]
    
    if dataset_data.empty:
        print(f"No data available for {selected_dataset}. Skipping plot.")
        plt.close()
        return
    
    max_sample_size = dataset_data['Sample Size'].max() if not dataset_data.empty else 10
    
    for i, category in enumerate(categories):
        category_data = dataset_data[dataset_data['Category'] == category].sort_values('n_nouns')
        
        if len(category_data) > 1:
            x_values = category_data['n_nouns'].values
            y_values = category_data['Mean'].values
            ci_lowers = category_data['CI Lower'].values
            ci_uppers = category_data['CI Upper'].values
            sample_sizes = category_data['Sample Size'].values
            
            for j in range(len(x_values) - 1):
                line_width = get_line_width((sample_sizes[j] + sample_sizes[j+1])/2, max_sample_size)
                
                category_idx = categories.index(category) if category in categories else i
                category_color = category_colors[category_idx % len(category_colors)]
                
                ax.plot(x_values[j:j+2], y_values[j:j+2], 
                        color=category_color, 
                        linewidth=line_width)
                
                ax.fill_between(x_values[j:j+2], ci_lowers[j:j+2], ci_uppers[j:j+2], 
                                color=category_color, alpha=0.1)
                
                ax.plot(x_values[j:j+2], ci_lowers[j:j+2], 
                        color=category_color, 
                        linestyle='--', linewidth=0.8, alpha=1)
                ax.plot(x_values[j:j+2], ci_uppers[j:j+2], 
                        color=category_color, 
                        linestyle='--', linewidth=0.8, alpha=1)
        
        elif len(category_data) == 1:
            x_val = category_data['n_nouns'].values[0]
            y_val = category_data['Mean'].values[0]
            ci_low = category_data['CI Lower'].values[0]
            ci_high = category_data['CI Upper'].values[0]
            sample_size = category_data['Sample Size'].values[0]
            
            category_idx = categories.index(category) if category in categories else i
            category_color = category_colors[category_idx % len(category_colors)]
            
            ax.scatter(x_val, y_val, 
                      color=category_color, 
                      s=get_line_width(sample_size, max_sample_size) * 15)  
            
            ax.errorbar(x_val, y_val, 
                       yerr=[[y_val - ci_low], [ci_high - y_val]],
                       color=category_color, 
                       capsize=4, alpha=0.5, linewidth=0.8)  
    
    ax.set_ylim(0, 1)
    ax.set_xlabel('Number of Actors', fontsize=18, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1.3)
    ax.grid(True, linestyle='-', linewidth=0.5, color='lightgrey')
    
    if selected_dataset == 'Valid Comp':
        ax.set_xticks(range(9, 21))
    elif selected_dataset == 'Test':
        ax.set_xticks(range(21, 31))
    else:  # Train or Valid A
        ax.set_xticks(range(2, 9))
    
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('#dddddd')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_color('#dddddd')
    ax.spines['right'].set_linewidth(0.5)
    ax.grid(True, color='#dddddd')
    
    title = f'{MODEL_NAME} - {DATASET_NAME}: Compositional generalisation per density - {selected_dataset}'    
    ax.set_title(title, loc='center', fontsize=20, fontweight='bold')
    
    category_legend_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                                      markersize=16, markerfacecolor=col) 
                               for cat, col in zip(categories, category_colors[:len(categories)])]
    first_legend = ax.legend(handles=category_legend_handles, 
                             title='Density', 
                             loc='lower left', 
                             framealpha=0.5, 
                             edgecolor='grey', 
                             fancybox=False, 
                             fontsize='15')
    first_legend.get_title().set_fontweight('bold')
    first_legend.get_title().set_fontsize('15')
    
    sample_sizes_to_show = [50, 100, 150, 200, 250]
    max_size = max_sample_size
    sample_sizes_to_show = [s for s in sample_sizes_to_show if s <= max_size * 2]
    if not sample_sizes_to_show:
        sample_sizes_to_show = [max(10, int(max_size / 3)), max(20, int(max_size / 2)), max(30, int(max_size))]
    
    sample_size_lines = [Line2D([], [], 
                               color='grey', 
                               linewidth=get_line_width(size, max_sample_size), 
                               solid_capstyle='butt') 
                        for size in sample_sizes_to_show]
    
    second_legend = ax.legend(handles=sample_size_lines, 
                             labels=[str(size) for size in sample_sizes_to_show],
                             title='Samples', 
                             loc='lower right', 
                             framealpha=0.5, 
                             edgecolor='grey', 
                             fancybox=False, 
                             fontsize='15')
    second_legend.get_title().set_fontweight('bold')
    second_legend.get_title().set_fontsize('15')
    
    ax.add_artist(first_legend)
    
    plt.tight_layout()
    safe_dataset_name = selected_dataset.lower().replace(" ", "_")
    plt.savefig(f'{MODEL_NAME.lower()}_{DATASET_NAME}_{safe_dataset_name}_category_breakdown.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_analysis(df_plot, categories, category_colors, datasets, dataset_colors,
                               train_accuracies, validation_accuracies, test_accuracies, validb_accuracies, 
                               max_sample_size):
    """Create comprehensive visualization with all datasets and categories."""
    plt.figure(figsize=(17, 8))
    ax = plt.gca()
    
    if df_plot.empty:
        print("No data available for comprehensive analysis. Skipping plot.")
        plt.close()
        return
    
    x_regions = {
        'Train': (df_plot[df_plot['Set'] == 'Train']['n_nouns'].min() if not df_plot[df_plot['Set'] == 'Train'].empty else 2,
                 df_plot[df_plot['Set'] == 'Train']['n_nouns'].max() if not df_plot[df_plot['Set'] == 'Train'].empty else 8),
        'Valid A': (df_plot[df_plot['Set'] == 'Valid A']['n_nouns'].min() if not df_plot[df_plot['Set'] == 'Valid A'].empty else 2,
                   df_plot[df_plot['Set'] == 'Valid A']['n_nouns'].max() if not df_plot[df_plot['Set'] == 'Valid A'].empty else 8),
        'Valid Comp': (df_plot[df_plot['Set'] == 'Valid Comp']['n_nouns'].min() if not df_plot[df_plot['Set'] == 'Valid Comp'].empty else 9,
                      df_plot[df_plot['Set'] == 'Valid Comp']['n_nouns'].max() if not df_plot[df_plot['Set'] == 'Valid Comp'].empty else 20),
        'Test': (df_plot[df_plot['Set'] == 'Test']['n_nouns'].min() if not df_plot[df_plot['Set'] == 'Test'].empty else 21,
                df_plot[df_plot['Set'] == 'Test']['n_nouns'].max() if not df_plot[df_plot['Set'] == 'Test'].empty else 30)
    }
    
    for dataset, (min_val, max_val) in x_regions.items():
        if min_val > max_val:
            if dataset in ['Train', 'Valid A']:
                x_regions[dataset] = (2, 8)
            elif dataset == 'Valid Comp':
                x_regions[dataset] = (9, 20)
            else:  
                x_regions[dataset] = (21, 30)
    
    dataset_means = {}
    for dataset in datasets:
        dataset_data = df_plot[df_plot['Set'] == dataset]
        if not dataset_data.empty:
            weights = dataset_data['Sample Size']
            weighted_mean = np.average(dataset_data['Mean'], weights=weights) if weights.sum() > 0 else 0.5
            dataset_means[dataset] = weighted_mean
        else:
            dataset_means[dataset] = 0.5
    
    for dataset, (min_val, max_val) in x_regions.items():
        if min_val < max_val and dataset in datasets:  
            range_vals = np.linspace(min_val, max_val, 100)
            dataset_idx = datasets.index(dataset)
            dataset_color = dataset_colors[dataset_idx % len(dataset_colors)]
            
            ax.plot(range_vals, [dataset_means.get(dataset, 0.5)] * len(range_vals), 
                   color=dataset_color, linewidth=1.0, linestyle='--', alpha=0.7, 
                   label=f'{dataset} Avg')
    
    for set_name in datasets:
        dataset_data = df_plot[df_plot['Set'] == set_name]
        dataset_idx = datasets.index(set_name)
        dataset_color = dataset_colors[dataset_idx % len(dataset_colors)]
        
        for category in categories:
            category_data = dataset_data[dataset_data['Category'] == category].sort_values('n_nouns')
            
            if len(category_data) <= 1:
                continue  
                
            category_idx = categories.index(category) if category in categories else 0
            category_color = category_colors[category_idx % len(category_colors)]
            
            for j in range(len(category_data) - 1):
                current = category_data.iloc[j]
                next_point = category_data.iloc[j + 1]
                
                width = get_line_width(current['Sample Size'], max_sample_size)
                
                ax.plot([current['n_nouns'], next_point['n_nouns']], 
                        [current['Mean'], next_point['Mean']], 
                        color=category_color, 
                        linewidth=width)
                
                ax.fill_between([current['n_nouns'], next_point['n_nouns']], 
                                [current['CI Lower'], next_point['CI Lower']], 
                                [current['CI Upper'], next_point['CI Upper']], 
                                color=category_color, alpha=0.1)
                
                ax.plot([current['n_nouns'], next_point['n_nouns']], 
                        [current['CI Lower'], next_point['CI Lower']], 
                        color=category_color, linestyle='--', linewidth=0.8, alpha=1)  # Reduced from 1.0
                
                ax.plot([current['n_nouns'], next_point['n_nouns']], 
                        [current['CI Upper'], next_point['CI Upper']], 
                        color=category_color, linestyle='--', linewidth=0.8, alpha=1)  # Reduced from 1.0
    
    all_x_values = df_plot['n_nouns'].dropna()
    if len(all_x_values) > 0:
        min_x = max(1.5, all_x_values.min() - 1)
        max_x = min(30.5, all_x_values.max() + 1)
    else:
        min_x, max_x = 1.5, 30.5
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='lightgrey')
    ax.set_xlim(left=min_x, right=max_x)
    ax.set_xticks(range(int(min_x) + 1, int(max_x)))
    ax.set_xlabel('Number of Actors', fontsize=15, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=15, fontweight='bold')
    ax.set_ylim(bottom=0, top=1)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_color('#dddddd')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_color('#dddddd')
    ax.spines['right'].set_linewidth(0.5)
    ax.grid(True, color='#dddddd')
    
    ax.set_title(f"{MODEL_NAME} - {DATASET_NAME}: Compositional generalisation", 
            pad=10, fontsize=18, fontweight='bold')
    ax.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1)
    
    category_handles = [plt.Line2D([], [], marker='o', color=category_colors[i % len(category_colors)],
                                 linestyle='', markersize=10)
                      for i in range(len(categories))]
    
    category_legend = ax.legend(handles=category_handles,
                              labels=categories,
                              title='Density',
                              loc='upper left',
                              frameon=True,
                              framealpha=0.5,
                              edgecolor='grey',
                              fancybox=False,
                              title_fontsize='large',
                              fontsize=15)
    category_legend.get_title().set_fontweight('bold')
    plt.gca().add_artist(category_legend)
    
    sample_size_to_show = [50, 100, 150, 200, 250]
    actual_max = df_plot['Sample Size'].max() if not df_plot.empty else 50
    sample_size_to_show = [s for s in sample_size_to_show if s <= 2 * actual_max]
    if not sample_size_to_show:
        sample_size_to_show = [int(actual_max/4), int(actual_max/2), int(actual_max)]
    
    legend_elements = [mpl.lines.Line2D([], [], 
                                      color='grey', 
                                      linewidth=get_line_width(lw, max_sample_size), 
                                      solid_capstyle='butt') 
                      for lw in sample_size_to_show]
    
    second_legend = ax.legend(handles=legend_elements, 
                             labels=[str(size) for size in sample_size_to_show], 
                             title='Samples', 
                             loc='lower right', 
                             frameon=True, 
                             framealpha=0.5, 
                             edgecolor='grey', 
                             fancybox=False,
                             title_fontsize='large', 
                             fontsize=15)
    second_legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'{MODEL_NAME.lower()}_{DATASET_NAME}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_evaluation_visualizations(data_info, eval_results, model_name="LSTM", dataset_name="all_2dir"):
    """
    Create comprehensive visualizations based on model evaluation results
    
    Args:
        data_info: Dictionary containing dataset information
        eval_results: Dictionary containing evaluation results
        model_name: Name of the model (e.g., "LSTM", "transformer")
        dataset_name: Name of the dataset (e.g., "all_2dir")
    """
    print(f"\n==== VERIFICATION OF DATA FOR VISUALIZATION ====")
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    
    df_keys = [k for k in data_info.keys() if k.endswith('_df')]
    print(f"DataFrames in data_info: {df_keys}")
    
    for key in df_keys:
        if key in data_info and isinstance(data_info[key], pd.DataFrame):
            df = data_info[key]
            print(f"  DataFrame '{key}': {len(df)} rows")
            print(f"    Columns: {list(df.columns)}")
            
            if 'category' in df.columns:
                print(f"    Categories: {sorted(df['category'].unique())}")
            else:
                print(f"    WARNING: No 'category' column!")
            
            actor_col = get_actor_column(df)
            if actor_col:
                print(f"    Actor column: '{actor_col}' with values: {sorted(df[actor_col].unique())}")
            else:
                print(f"    WARNING: No actor count column found!")
    
    print(f"\nEvaluation results structure:")
    if isinstance(eval_results, dict):
        print(f"  Keys: {list(eval_results.keys())}")
        if 'accuracy' in eval_results:
            print(f"  Overall accuracy: {eval_results['accuracy']}")
            
        for key in ['train_accuracies', 'validation_accuracies', 'validb_accuracies', 'test_accuracies']:
            if key in eval_results and eval_results[key]:
                print(f"  {key}: {list(eval_results[key].keys())}")
                for category, counts in eval_results[key].items():
                    print(f"    Category '{category}': {len(counts)} actor counts")
    else:
        print(f"  WARNING: eval_results is not a dictionary! Type: {type(eval_results)}")
    print("=================================================\n")
    
    print(f"Initializing visualization for {model_name.upper()} model on {dataset_name} dataset...")
    
    if model_name.lower() == "lstm":
        display_model = "LSTM"
    elif model_name.lower() == "transformer":
        display_model = "Transformer"
    else:
        display_model = model_name
        
    global MODEL_NAME, DATASET_NAME
    MODEL_NAME = display_model
    DATASET_NAME = dataset_name
    
    available_datasets = []
    if 'train_df' in data_info: available_datasets.append('Train')
    if 'val_df' in data_info: available_datasets.append('Valid A')
    if 'valb_df' in data_info: available_datasets.append('Valid Comp')
    if 'test_df' in data_info: available_datasets.append('Test')
    
    if not available_datasets:
        available_datasets = DATASET_SPLITS
    
    datasets = available_datasets
    print(f"Available datasets: {datasets}")
    
    dataset_colors = []
    for dataset in datasets:
        if dataset in DATASET_SPLITS:
            idx = DATASET_SPLITS.index(dataset)
            dataset_colors.append(DATASET_COLORS[idx])
        else:
            dataset_colors.append("#999999")
    
    detected_categories = get_categories(data_info)
    print(f"Detected density categories: {detected_categories}")
    
    categories = []
    category_colors = []
    
    for category in detected_categories:
        if category in DENSITY_CATEGORIES:
            idx = DENSITY_CATEGORIES.index(category)
            categories.append(category)
            category_colors.append(DENSITY_COLORS[idx])
        else:
            categories.append(category)
            if len(categories) <= len(DENSITY_COLORS):
                category_colors.append(DENSITY_COLORS[len(categories)-1])
            else:
                color_idx = len(categories) - len(DENSITY_COLORS) - 1
                cmap = cm.get_cmap('tab20', 20)
                category_colors.append(mpl.colors.rgb2hex(cmap(color_idx % 20)))
    
    if not categories:
        categories = DENSITY_CATEGORIES
        category_colors = DENSITY_COLORS
    
    print(f"Using categories: {categories}")
    print(f"Using category colors: {category_colors}")
    
    print("\nExtracting sample sizes...")
    sample_sizes = extract_sample_sizes(data_info)
    print(f"Sample sizes structure: {len(sample_sizes)} datasets")
    for dataset, category_data in sample_sizes.items():
        print(f"  {dataset}: {len(category_data)} categories")
        for category, sizes in category_data.items():
            print(f"    {category}: {len(sizes)} actor counts, sizes: {list(sizes.items())[0:3] if sizes else [] if sizes else []}...")
    
    print("\nExtracting accuracy data...")
    train_accuracies, validation_accuracies, test_accuracies, validb_accuracies = extract_accuracy_data(
        data_info, eval_results)
    
    print("Extracted accuracy data:")
    for name, acc_dict in [
        ("Train", train_accuracies),
        ("Valid A", validation_accuracies),
        ("Valid Comp", validb_accuracies),
        ("Test", test_accuracies)
    ]:
        if acc_dict:
            print(f"  {name}: {len(acc_dict)} categories")
            for category, counts in acc_dict.items():
                print(f"    {category}: {len(counts)} actor counts, example: {list(counts.items())[:2]}")
        else:
            print(f"  {name}: Empty")
    
    all_sample_sizes = [size for dataset in sample_sizes.values() 
                       for category in dataset.values() 
                       for size in category.values()]
    max_sample_size = max(all_sample_sizes) if all_sample_sizes else 1
    print(f"Maximum sample size: {max_sample_size}")
    
    print("\nPreparing plot data...")
    df_plot = prepare_plot_data(train_accuracies, validation_accuracies, test_accuracies, 
                               validb_accuracies, sample_sizes)
    print(f"Prepared DataFrame with {len(df_plot)} rows and columns: {list(df_plot.columns)}")
    
    if not df_plot.empty:
        print("Sample data (first 3 rows):")
        print(df_plot.head(3))
        
        print("\nCounts by Set and Category:")
        try:
            set_category_counts = df_plot.groupby(['Set', 'Category']).size().reset_index(name='count')
            print(set_category_counts)
        except Exception as e:
            print(f"Error counting by Set and Category: {e}")
    
    average_accuracy = []
    min_ci = []
    max_ci = []
    
    print("\nCalculating dataset statistics...")
    for dataset in datasets:
        dataset_data = df_plot[df_plot['Set'] == dataset]
        if not dataset_data.empty:
            weights = dataset_data['Sample Size']
            if weights.sum() > 0:
                weighted_acc = np.average(dataset_data['Mean'], weights=weights)
                dataset_min_ci = dataset_data['CI Lower'].min()
                dataset_max_ci = dataset_data['CI Upper'].max()
            else:
                weighted_acc, dataset_min_ci, dataset_max_ci = 0.5, 0.0, 1.0
        else:
            weighted_acc, dataset_min_ci, dataset_max_ci = 0.5, 0.0, 1.0
            
        average_accuracy.append(weighted_acc)
        min_ci.append(dataset_min_ci)
        max_ci.append(dataset_max_ci)
        
        print(f"  {dataset}: Acc={weighted_acc:.4f}, CI=[{dataset_min_ci:.4f}, {dataset_max_ci:.4f}]")
    
    print("\nCreating dataset summary plot...")
    plot_dataset_summary(datasets, dataset_colors, average_accuracy, min_ci, max_ci)

    print("\nCreating category breakdown plots for each dataset...")
    for dataset in datasets:
        try:
            print(f"  Processing {dataset}...")
            plot_category_breakdown(df_plot, categories, category_colors, dataset)
        except Exception as e:
            print(f"  Error creating category breakdown for {dataset}: {e}")

    print("\nCreating comprehensive analysis plot...")
    try:
        plot_comprehensive_analysis(df_plot, categories, category_colors, datasets, dataset_colors,
                                   train_accuracies, validation_accuracies, test_accuracies, 
                                   validb_accuracies, max_sample_size)
    except Exception as e:
        print(f"Error creating comprehensive analysis plot: {e}")
        import traceback
        traceback.print_exc()

    print("\nAll visualizations complete.")