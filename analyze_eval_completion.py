#!/usr/bin/env python3
"""Analyze eval output directory to identify completed, incomplete, and skipped houses.

This script scans an evaluation output directory and compares it against the benchmark
to identify:
1. Fully completed houses (all episodes done)
2. Incomplete houses (some but not all episodes done)
3. Not started houses (no episodes done)

Usage:
    python analyze_eval_completion.py --benchmark_dir /path/to/benchmark --eval_dir /path/to/eval_output
"""

import argparse
from collections import defaultdict
from pathlib import Path

from molmo_spaces.evaluation.benchmark_schema import load_all_episodes


def count_completed_episodes_per_house(eval_dir: Path) -> dict[int, int]:
    """Count how many episodes have been completed for each house.

    Args:
        eval_dir: Path to a timestamped eval output directory.

    Returns:
        Dict mapping house_id to count of completed episodes (with HDF5 files).
    """
    house_episode_counts = defaultdict(int)

    if not eval_dir.exists():
        return house_episode_counts

    for house_dir in eval_dir.iterdir():
        if not house_dir.is_dir() or not house_dir.name.startswith("house_"):
            continue

        try:
            house_id = int(house_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Count trajectory files in this house
        trajectory_files = list(house_dir.glob("trajectories*.h5"))
        house_episode_counts[house_id] = len(trajectory_files)

    return house_episode_counts


def count_expected_episodes_per_house(benchmark_dir: Path) -> dict[int, int]:
    """Count how many episodes each house should have according to the benchmark.

    Args:
        benchmark_dir: Path to JSON benchmark directory.

    Returns:
        Dict mapping house_id to expected number of episodes.
    """
    episodes = load_all_episodes(benchmark_dir)
    house_episode_counts = defaultdict(int)

    for episode in episodes:
        house_episode_counts[episode.house_index] += 1

    return house_episode_counts


def analyze_completion_status(
    expected_counts: dict[int, int],
    completed_counts: dict[int, int],
) -> tuple[list[int], list[tuple[int, int, int]], list[int]]:
    """Analyze which houses are complete, incomplete, or not started.

    Args:
        expected_counts: Dict mapping house_id to expected episode count
        completed_counts: Dict mapping house_id to completed episode count

    Returns:
        Tuple of (complete_houses, incomplete_houses, not_started_houses)
        - complete_houses: List of house IDs that are fully complete
        - incomplete_houses: List of (house_id, completed, expected) tuples
        - not_started_houses: List of house IDs with no completed episodes
    """
    complete_houses = []
    incomplete_houses = []
    not_started_houses = []

    all_house_ids = set(expected_counts.keys())

    for house_id in sorted(all_house_ids):
        expected = expected_counts[house_id]
        completed = completed_counts.get(house_id, 0)

        if completed == 0:
            not_started_houses.append(house_id)
        elif completed < expected:
            incomplete_houses.append((house_id, completed, expected))
        else:
            complete_houses.append(house_id)

    return complete_houses, incomplete_houses, not_started_houses


def print_analysis_report(
    complete_houses: list[int],
    incomplete_houses: list[tuple[int, int, int]],
    not_started_houses: list[int],
    total_expected_episodes: int,
    total_completed_episodes: int,
) -> None:
    """Print a formatted analysis report."""
    print("=" * 80)
    print("EVALUATION COMPLETION ANALYSIS")
    print("=" * 80)
    print()

    print(f"Overall Progress: {total_completed_episodes}/{total_expected_episodes} episodes "
          f"({100 * total_completed_episodes / total_expected_episodes:.1f}%)")
    print()

    print(f"Houses Summary:")
    print(f"  Complete:     {len(complete_houses)}")
    print(f"  Incomplete:   {len(incomplete_houses)}")
    print(f"  Not started:  {len(not_started_houses)}")
    print(f"  Total:        {len(complete_houses) + len(incomplete_houses) + len(not_started_houses)}")
    print()

    if complete_houses:
        print(f"Complete houses ({len(complete_houses)}):")
        print(f"  {complete_houses}")
        print()

    if incomplete_houses:
        print(f"Incomplete houses ({len(incomplete_houses)}):")
        for house_id, completed, expected in incomplete_houses:
            missing = expected - completed
            print(f"  house_{house_id}: {completed}/{expected} episodes "
                  f"({missing} missing, {100 * completed / expected:.1f}% done)")
        print()

    if not_started_houses:
        print(f"Not started houses ({len(not_started_houses)}):")
        print(f"  {not_started_houses}")
        print()

    print("=" * 80)


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze eval completion status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        required=True,
        help="Path to JSON benchmark directory",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Path to eval output directory (the timestamped directory containing house_* subdirs)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    benchmark_dir = Path(args.benchmark_dir).resolve()
    eval_dir = Path(args.eval_dir).resolve()

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    print(f"Analyzing benchmark: {benchmark_dir}")
    print(f"Against eval output: {eval_dir}")
    print()

    # Count expected episodes per house
    expected_counts = count_expected_episodes_per_house(benchmark_dir)
    total_expected = sum(expected_counts.values())

    # Count completed episodes per house
    completed_counts = count_completed_episodes_per_house(eval_dir)
    total_completed = sum(completed_counts.values())

    # Analyze completion status
    complete_houses, incomplete_houses, not_started_houses = analyze_completion_status(
        expected_counts, completed_counts
    )

    # Print report
    print_analysis_report(
        complete_houses,
        incomplete_houses,
        not_started_houses,
        total_expected,
        total_completed,
    )

    # Optionally write incomplete houses to a file for easy resumption
    if incomplete_houses or not_started_houses:
        output_file = eval_dir / "incomplete_analysis.txt"
        with open(output_file, "w") as f:
            f.write("Houses that need (re)processing:\n\n")

            if incomplete_houses:
                f.write("Incomplete houses (need to process remaining episodes):\n")
                for house_id, completed, expected in incomplete_houses:
                    f.write(f"  house_{house_id}: {completed}/{expected} episodes\n")
                f.write("\n")

            if not_started_houses:
                f.write("Not started houses:\n")
                f.write(f"  {not_started_houses}\n")

        print(f"Detailed analysis written to: {output_file}")


if __name__ == "__main__":
    main()
