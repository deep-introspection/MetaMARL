#!/usr/bin/env python3
"""Standalone script for visualizing fishery experiment results.

This script can be used to generate visualizations from existing experiment data,
allowing you to analyze fish/algae dynamics and fishermen actions after experiments
have been completed.

Usage examples:
  # List all available experiments
  python visualize_experiments.py --list
  
  # Visualize latest experiment automatically  
  python visualize_experiments.py --latest --summary-only
  
  # Visualize specific experiment iteration
  python visualize_experiments.py --experiment-dir runs/bilevel_20250906_143022 --iteration 0
  
  # Visualize all iterations with trajectory data from latest experiment
  python visualize_experiments.py --latest --all-iterations
  
  # Customize sustainability threshold
  python visualize_experiments.py --latest --iteration 0 --threshold 0.15
"""

import argparse
import os
import sys
import glob
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt

from visualization import (
    load_and_visualize_experiment, 
    create_experiment_summary_visualization
)


def find_latest_experiment() -> Optional[str]:
    """Find the most recent experiment directory in runs/.
    
    Returns:
        Path to latest experiment directory, or None if none found
    """
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return None
    
    # Look for directories matching timestamped patterns
    patterns = [
        "runs/bilevel_*",          # bilevel_YYYYMMDD_HHMMSS
        "runs/experiment_*",       # experiment_YYYYMMDD_HHMMSS
        "runs/fine_tuning_*",      # fine_tuning_YYYYMMDD_HHMMSS
        "runs/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]", # Any with timestamp suffix
    ]
    
    experiment_dirs = []
    for pattern in patterns:
        experiment_dirs.extend(glob.glob(pattern))
    
    # Also check for legacy bilevel_logs directory
    if os.path.exists("runs/bilevel_logs"):
        experiment_dirs.append("runs/bilevel_logs")
    
    if not experiment_dirs:
        return None
    
    # Sort by modification time (most recent first)
    experiment_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return experiment_dirs[0]


def list_available_experiments() -> None:
    """Print available experiment directories."""
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print("No runs/ directory found")
        return
    
    # Find all potential experiment directories
    all_dirs = []
    for item in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, item)
        if os.path.isdir(full_path):
            all_dirs.append(full_path)
    
    if not all_dirs:
        print("No experiment directories found in runs/")
        return
    
    # Sort by modification time (most recent first)
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print("Available experiment directories:")
    for i, exp_dir in enumerate(all_dirs):
        mtime = os.path.getmtime(exp_dir)
        mtime_str = os.popen(f'stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "{exp_dir}"').read().strip()
        marker = " (latest)" if i == 0 else ""
        print(f"  {exp_dir} - modified {mtime_str}{marker}")
    print(f"\nUse --latest to automatically select: {all_dirs[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize fishery experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments (made optional to support --latest)
    parser.add_argument("--experiment-dir", type=str,
                       help="Path to experiment directory containing results")
    parser.add_argument("--latest", action="store_true",
                       help="Automatically find and visualize the latest experiment in runs/")
    parser.add_argument("--list", action="store_true",
                       help="List all available experiment directories and exit")
    
    # What to visualize
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--iteration", type=int,
                      help="Specific outer iteration to visualize")
    group.add_argument("--all-iterations", action="store_true",
                      help="Visualize all iterations with trajectory data")
    group.add_argument("--summary-only", action="store_true",
                      help="Generate only the experiment summary visualization")
    
    # Options
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Fish population collapse threshold for visualization")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for plots (defaults to experiment-dir)")
    parser.add_argument("--show-plots", action="store_true",
                       help="Display plots interactively (requires display)")
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        list_available_experiments()
        sys.exit(0)
    
    # Handle --latest flag
    if args.latest:
        if args.experiment_dir:
            print("Error: Cannot use both --latest and --experiment-dir")
            sys.exit(1)
        args.experiment_dir = find_latest_experiment()
        if not args.experiment_dir:
            print("Error: No experiments found in runs/ directory")
            print("Try running: python visualize_experiments.py --list")
            sys.exit(1)
        print(f"Using latest experiment: {args.experiment_dir}")
    
    # Validate experiment directory
    if not args.experiment_dir:
        print("Error: Must specify either --experiment-dir or --latest")
        sys.exit(1)
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir or args.experiment_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set matplotlib backend for display
    if args.show_plots:
        try:
            matplotlib.use('TkAgg')
            print("Using interactive display for plots")
        except ImportError:
            print("Warning: Interactive display not available, saving plots only")
            args.show_plots = False
    
    success_count = 0
    total_attempts = 0
    
    try:
        # Generate experiment summary
        if not args.iteration and not args.all_iterations:
            print("Generating experiment summary visualization...")
            total_attempts += 1
            try:
                summary_fig = create_experiment_summary_visualization(
                    args.experiment_dir,
                    save_path=os.path.join(output_dir, "experiment_summary.png")
                )
                print(f"✓ Summary visualization saved: {os.path.join(output_dir, 'experiment_summary.png')}")
                
                if args.show_plots:
                    plt.show()
                else:
                    plt.close(summary_fig)
                
                success_count += 1
                
            except Exception as e:
                print(f"✗ Could not generate summary visualization: {e}")
            
            if args.summary_only:
                print(f"\nCompleted: {success_count}/{total_attempts} visualizations generated successfully")
                return
        
        # Find available iterations with trajectory data
        available_iterations = []
        if os.path.exists(args.experiment_dir):
            for item in os.listdir(args.experiment_dir):
                if item.startswith("outer_") and os.path.isdir(os.path.join(args.experiment_dir, item)):
                    try:
                        iter_num = int(item.split("_")[1])
                        # Check if trajectory data exists
                        iter_dir = os.path.join(args.experiment_dir, item)
                        has_traces = any(f.startswith("best_trace_") and f.endswith(".json") 
                                       for f in os.listdir(iter_dir))
                        if has_traces:
                            available_iterations.append(iter_num)
                    except (ValueError, IndexError):
                        continue
        
        available_iterations.sort()
        
        if not available_iterations:
            print("No trajectory data found in experiment directory.")
            print("Run experiments with --trace-episodes > 0 to generate trajectory data for visualization.")
            if success_count == 0:
                sys.exit(1)
            return
        
        print(f"Found trajectory data for iterations: {available_iterations}")
        
        # Determine which iterations to process
        iterations_to_process = []
        if args.iteration is not None:
            if args.iteration in available_iterations:
                iterations_to_process = [args.iteration]
            else:
                print(f"Error: Iteration {args.iteration} not found or has no trajectory data")
                print(f"Available iterations: {available_iterations}")
                if success_count == 0:
                    sys.exit(1)
                return
        elif args.all_iterations:
            iterations_to_process = available_iterations
        else:
            # Default: process the last (best) iteration
            iterations_to_process = [available_iterations[-1]]
        
        # Generate trajectory visualizations
        for iteration in iterations_to_process:
            print(f"\nProcessing iteration {iteration}...")
            total_attempts += 3  # Each iteration generates 3 types of plots
            
            try:
                # Override output paths to use specified output directory
                iter_output_dir = os.path.join(output_dir, f"outer_{iteration}")
                os.makedirs(iter_output_dir, exist_ok=True)
                
                figures = load_and_visualize_experiment(
                    args.experiment_dir,
                    iteration,
                    sustainability_threshold=args.threshold,
                    save_plots=False  # We'll save manually to control paths
                )
                
                # Save figures to output directory
                for plot_type, fig in figures.items():
                    save_path = os.path.join(iter_output_dir, f"{plot_type}.png")
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"  ✓ {plot_type.capitalize()} plot saved: {save_path}")
                    
                    success_count += 1
                    
                    if args.show_plots:
                        plt.show()
                    else:
                        plt.close(fig)
                
            except FileNotFoundError as e:
                print(f"  ✗ Trajectory data not found for iteration {iteration}: {e}")
            except Exception as e:
                print(f"  ✗ Error generating plots for iteration {iteration}: {e}")
        
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    
    print(f"\nCompleted: {success_count}/{total_attempts} visualizations generated successfully")
    
    if output_dir != args.experiment_dir:
        print(f"Plots saved to: {output_dir}")
    
    # Summary of what was generated
    if success_count > 0:
        print("\nGenerated visualizations:")
        if os.path.exists(os.path.join(output_dir, "experiment_summary.png")):
            print(f"  - Experiment summary: {os.path.join(output_dir, 'experiment_summary.png')}")
        
        for iteration in iterations_to_process:
            iter_output_dir = os.path.join(output_dir, f"outer_{iteration}")
            if os.path.exists(iter_output_dir):
                print(f"  - Iteration {iteration} plots: {iter_output_dir}/")


if __name__ == "__main__":
    main()