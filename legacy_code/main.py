"""Main experiment runner for bilevel fishery optimization.

This script orchestrates the complete bilevel optimization experiment,
combining Evolution Strategies (outer loop) with Independent PPO (inner loop).
"""

# Set CPU-only environment variables BEFORE any other imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CUDA"] = "0" 
os.environ["RAY_USE_MPS"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import argparse
import time
from typing import List, Tuple

import numpy as np

# Optional dependency
try:
    import wandb
except ImportError:
    wandb = None

from config import DEFAULT_TRAINING_CONFIG, ES_CONFIG, LOGGING_CONFIG
from evaluation import EvaluationMetrics, evaluate_mechanism_with_metrics
from evolution_strategies import sample_es_population, update_es_parameters
from mechanism import MechanismParameters, map_unit_vector_to_mechanism
from training import build_ppo_algorithm, initialize_ray, shutdown_ray
from utils import (
    save_csv, save_json, initialize_wandb_logging, finalize_wandb_logging,
    create_results_summary, format_time_duration
)
from visualization import load_and_visualize_experiment, create_experiment_summary_visualization


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Bilevel optimization for sustainable fishery management",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment parameters
    env_group = parser.add_argument_group("Environment parameters")
    env_group.add_argument("--num-fishermen", type=int, default=DEFAULT_TRAINING_CONFIG["num_fishermen"],
                          help="Number of fishermen agents")
    env_group.add_argument("--horizon", type=int, default=DEFAULT_TRAINING_CONFIG["horizon"],
                          help="Episode length in timesteps")
    
    # Optimization parameters
    opt_group = parser.add_argument_group("Optimization parameters")
    opt_group.add_argument("--outer-iters", type=int, default=DEFAULT_TRAINING_CONFIG["outer_iters"],
                          help="Number of outer ES iterations")
    opt_group.add_argument("--inner-iters", type=int, default=DEFAULT_TRAINING_CONFIG["inner_iters"],
                          help="Number of inner PPO training iterations")
    opt_group.add_argument("--eval-episodes", type=int, default=DEFAULT_TRAINING_CONFIG["eval_episodes"],
                          help="Number of evaluation episodes per candidate")
    opt_group.add_argument("--pop-size", type=int, default=DEFAULT_TRAINING_CONFIG["pop_size"],
                          help="Population size for Evolution Strategies")
    
    # Learning parameters
    learn_group = parser.add_argument_group("Learning parameters")
    learn_group.add_argument("--sigma", type=float, default=ES_CONFIG["sigma"],
                            help="Initial noise standard deviation for ES")
    learn_group.add_argument("--lr-mean", type=float, default=ES_CONFIG["lr_mean"],
                            help="Learning rate for ES mean update")
    learn_group.add_argument("--lr-sigma", type=float, default=ES_CONFIG["lr_sigma"],
                            help="Learning rate for ES sigma adaptation")
    
    # System parameters
    sys_group = parser.add_argument_group("System parameters")
    sys_group.add_argument("--workers", type=int, default=DEFAULT_TRAINING_CONFIG["workers"],
                          help="Number of parallel workers for training")
    sys_group.add_argument("--seed", type=int, default=0,
                          help="Random seed for reproducibility")
    
    # Sustainability parameters
    sus_group = parser.add_argument_group("Sustainability parameters")
    sus_group.add_argument("--sustain-weight", type=float, default=DEFAULT_TRAINING_CONFIG["sustain_weight"],
                          help="Weight for sustainability penalty in objective")
    sus_group.add_argument("--sus-threshold", type=float, default=DEFAULT_TRAINING_CONFIG["sus_threshold"],
                          help="Fish stock level considered collapse threshold")
    
    # Logging parameters
    log_group = parser.add_argument_group("Logging parameters")
    log_group.add_argument("--log-dir", type=str, default=LOGGING_CONFIG["default_log_dir"],
                          help="Directory to store experiment logs")
    log_group.add_argument("--trace-episodes", type=int, default=0,
                          help="Number of episodes to trace for best candidate (0=disabled)")
    
    # W&B parameters
    wandb_group = parser.add_argument_group("Weights & Biases logging")
    wandb_group.add_argument("--wandb", action="store_true",
                            help="Enable Weights & Biases logging")
    wandb_group.add_argument("--wandb-project", type=str, default=LOGGING_CONFIG["wandb_project"],
                            help="W&B project name")
    wandb_group.add_argument("--wandb-entity", type=str, default=None,
                            help="W&B entity/user or team")
    wandb_group.add_argument("--wandb-run-name", type=str, default=None,
                            help="W&B run name")
    
    return parser


def _log_training_metrics(wandb_run, training_result: dict, training_iter: int, outer_iter: int, candidate_idx: int, global_step: int):
    """Log training metrics to W&B if enabled."""
    if wandb_run is not None:
        # Extract episode reward
        episode_reward = training_result.get("env_runners", {}).get("episode_reward_mean", float("nan"))

        # Build metrics dict
        metrics = {
            "train/episode_reward_mean": episode_reward,
            "train/inner_iter": training_iter,
            "train/outer_iter": outer_iter,
            "train/candidate_idx": candidate_idx,
        }

        # Add RLlib training metrics if available
        learner_stats = training_result.get("learners", {}).get("default_policy", {})
        if learner_stats:
            if "policy_loss" in learner_stats:
                metrics["train/policy_loss"] = learner_stats["policy_loss"]
            if "vf_loss" in learner_stats:
                metrics["train/vf_loss"] = learner_stats["vf_loss"]
            if "entropy" in learner_stats:
                metrics["train/entropy"] = learner_stats["entropy"]
            if "kl" in learner_stats:
                metrics["train/kl_divergence"] = learner_stats["kl"]

        # Log with warning if episode_reward is NaN
        if np.isnan(episode_reward):
            import warnings
            warnings.warn(f"Episode reward is NaN at iter {training_iter}, outer {outer_iter}, candidate {candidate_idx}")

        wandb.log(metrics, step=global_step)


def _log_evaluation_metrics(wandb_run, metrics: EvaluationMetrics, mechanism_params: MechanismParameters,
                           objective_score: float, outer_iter: int, candidate_idx: int, global_step: int):
    """Log evaluation metrics and mechanism parameters to W&B if enabled."""
    if wandb_run is not None:
        wandb.log({
            "eval/mean_reward": metrics.mean_reward,
            "eval/sustainability_penalty": metrics.sustainability_penalty,
            "eval/mean_min_fish": metrics.mean_min_fish,
            "eval/collapse_rate": metrics.collapse_rate,
            "eval/objective_score": objective_score,
            "eval/outer_iter": outer_iter,
            "eval/candidate_idx": candidate_idx,
            "mechanism/fixed_quota": mechanism_params.fixed_quota,
            "mechanism/prop_quota": mechanism_params.prop_quota,
            "mechanism/min_stock": mechanism_params.min_stock,
            "mechanism/fine_amount": mechanism_params.fine_amount,
            "mechanism/ban_period": mechanism_params.ban_period,
        }, step=global_step)


def train_and_evaluate_candidate(
    candidate_vector: np.ndarray,
    candidate_idx: int,
    args,
    outer_iter: int,
    wandb_run,
    global_step_counter: List[int]
) -> Tuple[float, MechanismParameters, EvaluationMetrics, List, float]:
    """Train and evaluate a single mechanism candidate.

    Args:
        candidate_vector: Unit vector representing mechanism parameters
        candidate_idx: Index of candidate in population
        args: Command-line arguments
        outer_iter: Current outer iteration number
        wandb_run: W&B run object (or None)
        global_step_counter: Mutable list with global step counter [step]

    Returns:
        Tuple of (objective_score, mechanism_params, metrics, training_progress, training_time)
    """
    import time
    candidate_start_time = time.time()
    # Map candidate to mechanism parameters
    mechanism_params = map_unit_vector_to_mechanism(candidate_vector)
    
    # Build PPO algorithm
    env_config = {"horizon": args.horizon}
    algorithm, dummy_env = build_ppo_algorithm(
        args.num_fishermen, mechanism_params, num_workers=args.workers, env_config=env_config
    )
    
    # Inner-loop training with periodic logging
    training_progress = []
    local_log_frequency = max(1, args.inner_iters // 20)  # Log locally more frequently
    wandb_log_frequency = max(1, args.inner_iters // 5)   # Log to W&B less frequently (avoid rate limits)

    for training_iter in range(args.inner_iters):
        training_result = algorithm.train()

        # Extract episode reward
        episode_reward = training_result.get("env_runners", {}).get("episode_reward_mean", float("nan"))

        # Save locally more frequently
        should_log_local = (training_iter % local_log_frequency == 0) or (training_iter == args.inner_iters - 1)
        if should_log_local:
            training_progress.append([training_iter, episode_reward])

        # Log to W&B less frequently
        should_log_wandb = (training_iter % wandb_log_frequency == 0) or (training_iter == args.inner_iters - 1)
        if should_log_wandb:
            global_step_counter[0] += 1
            _log_training_metrics(wandb_run, training_result, training_iter, outer_iter, candidate_idx, global_step_counter[0])
    
    # Evaluate with sustainability metrics
    metrics, _ = evaluate_mechanism_with_metrics(
        algorithm, dummy_env, 
        num_episodes=args.eval_episodes, 
        sustainability_threshold=args.sus_threshold, 
        record_trajectories=False
    )
    
    # Compute objective score
    objective_score = metrics.objective_score(args.sustain_weight)

    global_step_counter[0] += 1
    _log_evaluation_metrics(wandb_run, metrics, mechanism_params, objective_score, outer_iter, candidate_idx, global_step_counter[0])

    algorithm.stop()

    candidate_duration = time.time() - candidate_start_time
    return objective_score, mechanism_params, metrics, training_progress, candidate_duration


def run_bilevel_optimization(args) -> dict:
    """Run the complete bilevel optimization experiment.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with experiment results
    """
    experiment_start_time = time.time()
    
    # Create timestamped experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.log_dir == LOGGING_CONFIG["default_log_dir"]:
        # Use default with timestamp
        args.log_dir = f"runs/bilevel_{timestamp}"
    else:
        # Append timestamp to custom directory
        args.log_dir = f"{args.log_dir}_{timestamp}"
    
    # Ensure the timestamped directory exists
    from utils import ensure_directory
    ensure_directory(args.log_dir)
    
    print(f"Experiment results will be saved to: {args.log_dir}")
    
    # Initialize random generator and Ray
    rng = np.random.default_rng(args.seed)
    initialize_ray()
    
    # Initialize logging
    wandb_config = {
        "outer_iterations": args.outer_iters,
        "inner_iterations": args.inner_iters,
        "evaluation_episodes": args.eval_episodes,
        "population_size": args.pop_size,
        "initial_sigma": args.sigma,
        "mean_learning_rate": args.lr_mean,
        "sigma_learning_rate": args.lr_sigma,
        "sustainability_weight": args.sustain_weight,
        "sustainability_threshold": args.sus_threshold,
        "num_fishermen": args.num_fishermen,
        "num_workers": args.workers,
        "random_seed": args.seed,
        "horizon": args.horizon,
    }
    
    wandb_run = None
    if args.wandb:
        wandb_run = initialize_wandb_logging(
            args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            config=wandb_config
        )

    # Initialize Evolution Strategies distribution over θ ∈ [0,1]^5
    if hasattr(args, 'init_mean') and args.init_mean is not None:
        es_mean = np.array(args.init_mean, dtype=np.float32)  # Use custom initialization
        print(f"Initialized ES mean from custom values: {es_mean}")
    else:
        es_mean = np.full(5, 0.5, dtype=np.float32)  # Start at center of unit cube
        print(f"Initialized ES mean at center: {es_mean}")
    es_sigma = float(args.sigma)

    # Track best mechanism found across all iterations
    best_objective_score = -float("inf")
    best_unit_vector = es_mean.copy()

    experiment_summary = []  # Summary per outer iteration
    es_evolution = []  # Track ES mean and sigma over iterations
    global_step_counter = [0]  # Mutable counter for W&B step parameter

    print(f"Starting bilevel optimization with {args.outer_iters} outer iterations...")
    print(f"Configuration: {args.pop_size} candidates, {args.inner_iters} inner iters, {args.eval_episodes} eval episodes")
    
    for outer_iter in range(args.outer_iters):
        iteration_start_time = time.time()

        # Store previous mean for convergence tracking
        prev_es_mean = es_mean.copy()
        prev_es_sigma = es_sigma

        # Sample candidate mechanisms
        population = sample_es_population(es_mean, es_sigma, args.pop_size, rng)
        candidate_records = []  # Per-candidate data for this iteration
        objective_scores = []
        candidate_times = []  # Time per candidate

        # Track best candidate in this iteration
        best_candidate_idx = None
        best_iteration_score = -float("inf")
        best_iteration_vector = None

        print(f"\nOuter iteration {outer_iter + 1}/{args.outer_iters}:")
        print(f"  Evaluating {len(population)} mechanism candidates...")

        # Evaluate each candidate mechanism
        for candidate_idx, candidate_vector in enumerate(population):
            print(f"    Training candidate {candidate_idx + 1}/{len(population)}...", end=" ")

            # Train and evaluate this candidate
            objective_score, mechanism_params, eval_metrics, training_data, cand_time = train_and_evaluate_candidate(
                candidate_vector, candidate_idx, args, outer_iter, wandb_run, global_step_counter
            )
            
            objective_scores.append(objective_score)
            candidate_times.append(cand_time)

            # Save training progress for this candidate
            save_csv(["iter", "episode_reward_mean"], training_data,
                     os.path.join(args.log_dir, f"outer_{outer_iter}", f"train_cand_{candidate_idx}.csv"))

            # Record candidate data
            candidate_records.append([
                candidate_idx, candidate_vector.round(4).tolist(),
                mechanism_params.fixed_quota, mechanism_params.prop_quota,
                mechanism_params.min_stock, mechanism_params.fine_amount, mechanism_params.ban_period,
                eval_metrics.mean_reward, eval_metrics.sustainability_penalty,
                eval_metrics.mean_min_fish, eval_metrics.collapse_rate, objective_score,
                cand_time,
            ])

            # Track best candidate this iteration
            if objective_score > best_iteration_score:
                best_iteration_score = objective_score
                best_candidate_idx = candidate_idx
                best_iteration_vector = candidate_vector.copy()

            print(f"Score: {objective_score:.3f} ({format_time_duration(cand_time)})")

        # Save candidate evaluation results
        save_csv(
            ["cand_idx", "unit_vector", "fixed_quota", "prop_quota", "min_stock", "fine_amount", "ban_period",
             "mean_reward", "sustainability_penalty", "mean_min_fish", "collapse_rate", "objective_score", "time_seconds"],
            candidate_records,
            os.path.join(args.log_dir, f"outer_{outer_iter}", "candidates.csv"),
        )

        # Update Evolution Strategies parameters
        es_mean, es_sigma = update_es_parameters(
            current_mean=es_mean,
            current_sigma=es_sigma,
            population=population,
            fitness_scores=objective_scores,
            mean_learning_rate=args.lr_mean,
            sigma_learning_rate=args.lr_sigma
        )

        iteration_duration = time.time() - iteration_start_time
        mean_score = np.mean(objective_scores)
        std_score = np.std(objective_scores)
        min_score = np.min(objective_scores)
        max_score = np.max(objective_scores)

        # Compute convergence metrics
        delta_mean = np.linalg.norm(es_mean - prev_es_mean)
        delta_sigma = abs(es_sigma - prev_es_sigma)

        # Compute sustainability metrics
        num_sustainable = sum(1 for r in candidate_records if r[-3] < args.sus_threshold)  # collapse_rate column
        sustainability_rate = num_sustainable / len(candidate_records)

        # Track ES evolution
        es_evolution.append([
            outer_iter,
            *es_mean.tolist(),
            es_sigma,
            delta_mean,
            delta_sigma,
        ])

        print(f"\n  Results: mean={mean_score:.3f}, std={std_score:.3f}, best={best_iteration_score:.3f}, sigma={es_sigma:.3f}")
        print(f"  Convergence: Δmean={delta_mean:.4f}, Δsigma={delta_sigma:.4f}")
        print(f"  Sustainability: {num_sustainable}/{len(candidate_records)} ({sustainability_rate:.1%}) candidates sustainable")
        print(f"  Best mechanism: {map_unit_vector_to_mechanism(best_iteration_vector)}")
        print(f"  Iteration completed in {format_time_duration(iteration_duration)} (avg {format_time_duration(np.mean(candidate_times))}/candidate)")

        # Log iteration summary to W&B
        if wandb_run is not None:
            global_step_counter[0] += 1

            # Build metrics dict
            iteration_metrics = {
                "outer/mean_score": float(mean_score),
                "outer/std_score": float(std_score),
                "outer/min_score": float(min_score),
                "outer/max_score": float(max_score),
                "outer/best_iteration_score": float(best_iteration_score),
                "outer/sigma": float(es_sigma),
                "outer/iteration": outer_iter,
                "outer/iteration_duration": iteration_duration,
                "outer/avg_candidate_duration": float(np.mean(candidate_times)),
                "outer/delta_mean": float(delta_mean),
                "outer/delta_sigma": float(delta_sigma),
                "outer/sustainability_rate": float(sustainability_rate),
                "outer/num_sustainable": num_sustainable,
            }

            # Add ES mean components
            for i in range(5):
                iteration_metrics[f"es_mean/param_{i}"] = float(es_mean[i])

            # Add best mechanism from this iteration
            best_mech = map_unit_vector_to_mechanism(best_iteration_vector)
            iteration_metrics["best_iter_mechanism/fixed_quota"] = best_mech.fixed_quota
            iteration_metrics["best_iter_mechanism/prop_quota"] = best_mech.prop_quota
            iteration_metrics["best_iter_mechanism/min_stock"] = best_mech.min_stock
            iteration_metrics["best_iter_mechanism/fine_amount"] = best_mech.fine_amount
            iteration_metrics["best_iter_mechanism/ban_period"] = best_mech.ban_period

            wandb.log(iteration_metrics, step=global_step_counter[0])

        # Update global best if this iteration found a better mechanism
        if best_iteration_score > best_objective_score:
            best_objective_score = best_iteration_score
            best_unit_vector = best_iteration_vector.copy()
            print(f"  *** New best mechanism found! Score: {best_objective_score:.3f} ***")

        # Record trajectory traces for best candidate if requested
        if args.trace_episodes > 0 and best_candidate_idx is not None:
            print(f"  Recording {args.trace_episodes} trajectory episodes for best candidate...")
            
            # Re-train the best candidate to get fresh algorithm for tracing
            best_mechanism = map_unit_vector_to_mechanism(best_iteration_vector)
            env_config = {"horizon": args.horizon}
            trace_algorithm, trace_environment = build_ppo_algorithm(
                args.num_fishermen, best_mechanism, num_workers=args.workers, env_config=env_config
            )
            
            # Quick training
            for _ in range(min(20, args.inner_iters)):
                trace_algorithm.train()
            
            # Record trajectories with the trained algorithm
            _, trajectory_records = evaluate_mechanism_with_metrics(
                trace_algorithm, 
                trace_environment, 
                args.trace_episodes,
                args.sus_threshold,
                record_trajectories=True
            )
            
            # Save trajectory data
            if trajectory_records:
                trace_filename = f"best_trace_{outer_iter}.json"
                save_json(trajectory_records, os.path.join(args.log_dir, f"outer_{outer_iter}", trace_filename))
                
                # Also save as CSV for easier analysis
                trace_csv_filename = f"best_trace_{outer_iter}.csv"
                trace_csv_headers = ["episode", "step", "fish_population", "algae_population", 
                                   "total_harvest", "quota_limit"]
                trace_csv_data = [[r[h] for h in trace_csv_headers] for r in trajectory_records]
                save_csv(trace_csv_headers, trace_csv_data, 
                        os.path.join(args.log_dir, f"outer_{outer_iter}", trace_csv_filename))
                
                print(f"    Trajectories saved: {trace_filename}, {trace_csv_filename}")
            
            trace_algorithm.stop()
            del trace_algorithm, trace_environment

        # Save iteration summary
        experiment_summary.append([
            outer_iter, float(mean_score), float(std_score), float(min_score), float(max_score),
            best_iteration_score, best_iteration_vector.round(4).tolist(),
            map_unit_vector_to_mechanism(best_iteration_vector).to_dict(),
            iteration_duration, float(np.mean(candidate_times)),
            float(sustainability_rate), num_sustainable,
            float(delta_mean), float(delta_sigma)
        ])
        save_csv(
            ["outer_iter", "mean_score", "std_score", "min_score", "max_score",
             "best_iteration_score", "best_unit_vector", "best_mechanism_params",
             "duration_seconds", "avg_candidate_time", "sustainability_rate", "num_sustainable",
             "delta_mean", "delta_sigma"],
            experiment_summary,
            os.path.join(args.log_dir, "experiment_summary.csv")
        )

        # Save ES evolution
        save_csv(
            ["outer_iter", "mean_0", "mean_1", "mean_2", "mean_3", "mean_4", "sigma", "delta_mean", "delta_sigma"],
            es_evolution,
            os.path.join(args.log_dir, "es_evolution.csv")
        )

    experiment_duration = time.time() - experiment_start_time
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Best objective score achieved: {best_objective_score:.4f}")
    print(f"Total experiment time: {format_time_duration(experiment_duration)}")
    
    # Final training and checkpoint for globally best mechanism
    print("\nTraining final model with best mechanism...")
    final_mechanism = map_unit_vector_to_mechanism(best_unit_vector)
    env_config = {"horizon": args.horizon}
    final_algorithm, _ = build_ppo_algorithm(
        args.num_fishermen, final_mechanism, num_workers=args.workers, env_config=env_config
    )
    
    # Train final algorithm
    for iteration in range(args.inner_iters):
        final_algorithm.train()
        if (iteration + 1) % max(1, args.inner_iters // 5) == 0:
            print(f"  Final training progress: {iteration + 1}/{args.inner_iters}")
    
    checkpoint_path = final_algorithm.save()
    print(f"Model checkpoint saved: {checkpoint_path}")

    # Create comprehensive final results
    final_results = create_results_summary(
        experiment_config=wandb_config,
        best_mechanism=final_mechanism.to_dict(),
        best_score=best_objective_score,
        checkpoint_path=checkpoint_path,
        experiment_duration=experiment_duration
    )

    # Save final results
    results_path = os.path.join(args.log_dir, "experiment_results.json")
    save_json(final_results, results_path)
    print(f"\nExperiment results saved: {results_path}")
    
    # Generate experiment summary visualization
    try:
        print("Generating experiment summary visualization...")
        summary_fig = create_experiment_summary_visualization(
            args.log_dir,
            save_path=os.path.join(args.log_dir, "experiment_summary.png")
        )
        print(f"Summary visualization saved: {os.path.join(args.log_dir, 'experiment_summary.png')}")
    except Exception as e:
        print(f"Warning: Could not generate summary visualization: {e}")
    
    # Generate visualizations for iterations with trajectory data
    if args.trace_episodes > 0:
        print("Generating trajectory visualizations...")
        for outer_iter in range(args.outer_iters):
            try:
                figures = load_and_visualize_experiment(
                    args.log_dir, outer_iter, args.sus_threshold, save_plots=True
                )
                print(f"  Generated plots for iteration {outer_iter}: {list(figures.keys())}")
            except FileNotFoundError:
                # No trajectory data for this iteration, skip silently
                continue
            except Exception as e:
                print(f"  Warning: Could not generate plots for iteration {outer_iter}: {e}")
    
    print(f"\nBEST MECHANISM FOUND:")
    print(f"  {final_mechanism}")
    print(f"  Objective score: {best_objective_score:.4f}")

    # Finalize W&B logging
    if wandb_run is not None:
        summary_metrics = {
            "best_objective_score": float(best_objective_score),
            "best_mechanism_parameters": final_mechanism.to_dict(),
            "final_checkpoint": str(checkpoint_path),
            "total_outer_iterations": args.outer_iters,
            "experiment_duration_seconds": experiment_duration,
        }
        finalize_wandb_logging(wandb_run, summary_metrics)
        print("\nW&B logging completed.")

    # Clean up resources
    print("\nCleaning up Ray resources...")
    shutdown_ray()
    
    print("Bilevel optimization experiment completed successfully!")
    print(f"Results available in: {args.log_dir}")
    print(f"{'='*60}")
    
    return final_results


def main() -> None:
    """Main entry point for the bilevel optimization experiment."""
    # Suppress internal RLlib deprecation warnings for cleaner output
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    
    # Ensure CPU-only environment variables are set
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["USE_CUDA"] = "0" 
    os.environ["RAY_USE_MPS"] = "0"

    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Run the complete experiment
        results = run_bilevel_optimization(args)
        
        # Print final summary
        print(f"\nExperiment completed successfully!")
        print(f"Best objective score: {results['best_results']['objective_score']:.4f}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        shutdown_ray()
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        shutdown_ray()
        raise


if __name__ == "__main__":
    main()