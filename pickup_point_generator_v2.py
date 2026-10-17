"""Generate deterministic input files for the alternative-pickup-point model (V2).

The base test instances contain one node for each employee.  Model V2 gives
each employee two candidate pickup nodes:

* ``2*i - 1``: the employee's original (home) pickup node;
* ``2*i``: an alternative pickup node.

The original employee, dummy, and office distances are preserved exactly.
Distances involving newly introduced alternative pickup nodes are generated
deterministically from a user-supplied random seed.  The distance between an
employee's two pickup alternatives is set to 1000 so that a route cannot move
directly between the two representations of the same employee.

This script generates only the V2-specific files:

* ``Car_Owner/car_owner_testN_v2.csv``
* ``Distance_Matrix/distance_matrix_testN_v2.csv``

V2 uses the original employee-level ``schedule_testN.csv`` files without
expanding their rows.  Therefore, this script deliberately does not create or
overwrite schedule files.  The validated schedules should be stored separately
in the repository's ``Schedule`` directory.

Examples
--------
Generate all 24 instances from the repository's flat ``testproblems`` folder::

    python pickup_point_generator_v2.py \
        --input-dir ../testproblems \
        --output-dir . \
        --tests 1-24 \
        --overwrite

Generate only Tests 9 and 17::

    python pickup_point_generator_v2.py \
        --input-dir ../testproblems \
        --output-dir . \
        --tests 9,17 \
        --overwrite
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SEED = 123
DEFAULT_TESTS = tuple(range(1, 25))


@dataclass(frozen=True)
class BaseInstance:
    """Validated data loaded from one base (V1) test instance."""

    test_id: int
    employee_count: int
    dummy_node: int
    office_node: int
    car_owner: pd.Series
    schedule: pd.DataFrame
    distance_matrix: pd.DataFrame


def parse_test_ids(specification: str) -> list[int]:
    """Parse a comma-separated list containing IDs and inclusive ranges.

    Examples: ``"1-24"``, ``"9,17"``, and ``"1-4,9,17"``.
    """

    test_ids: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Invalid test range '{item}': start is greater than end."
                )
            test_ids.update(range(start, end + 1))
        else:
            test_ids.add(int(item))

    if not test_ids:
        raise argparse.ArgumentTypeError("At least one test ID must be provided.")
    if min(test_ids) < 1:
        raise argparse.ArgumentTypeError("Test IDs must be positive integers.")

    return sorted(test_ids)


def resolve_input_file(input_dir: Path, subdirectory: str, filename: str) -> Path:
    """Find an input file in either a flat or category-based data layout."""

    candidates = (
        input_dir / filename,
        input_dir / subdirectory / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Input file '{filename}' was not found. Searched:\n  - {searched}"
    )


def load_car_owner(path: Path) -> pd.Series:
    """Load a car-owner vector and return it with integer node indices."""

    raw = pd.read_csv(path, header=None)
    if raw.shape[1] == 1:
        values = pd.to_numeric(raw.iloc[:, 0], errors="raise").astype(float)
        values.index = range(len(values))
        return values

    if raw.shape[1] != 2:
        raise ValueError(
            f"{path}: expected one value column or an index/value pair, "
            f"but found {raw.shape[1]} columns."
        )

    node_ids = pd.to_numeric(raw.iloc[:, 0], errors="raise").astype(int)
    values = pd.to_numeric(raw.iloc[:, 1], errors="raise").astype(float)
    if node_ids.duplicated().any():
        raise ValueError(f"{path}: duplicate node IDs were found.")

    return pd.Series(values.to_numpy(), index=node_ids.to_numpy(), dtype=float)


def load_indexed_csv(path: Path) -> pd.DataFrame:
    """Load an indexed CSV and normalize row and column labels to integers."""

    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_numeric(frame.index, errors="raise").astype(int)
    frame.columns = pd.to_numeric(frame.columns, errors="raise").astype(int)
    return frame


def load_base_instance(input_dir: Path, test_id: int) -> BaseInstance:
    """Load and validate one base test instance."""

    schedule_path = resolve_input_file(
        input_dir, "Schedule", f"schedule_test{test_id}.csv"
    )
    car_owner_path = resolve_input_file(
        input_dir, "Car_Owner", f"car_owner_test{test_id}.csv"
    )
    distance_path = resolve_input_file(
        input_dir, "Distance_Matrix", f"distance_matrix_test{test_id}.csv"
    )

    schedule = load_indexed_csv(schedule_path).astype(int)
    distance = load_indexed_csv(distance_path).astype(float)
    car_owner = load_car_owner(car_owner_path)

    if distance.shape[0] != distance.shape[1]:
        raise ValueError(f"{distance_path}: the distance matrix must be square.")
    if list(distance.index) != list(distance.columns):
        raise ValueError(
            f"{distance_path}: row and column node labels must be identical."
        )
    if list(schedule.index) != list(distance.index):
        raise ValueError(
            f"Test {test_id}: schedule and distance-matrix node labels differ."
        )

    nodes = list(distance.index)
    expected_nodes = list(range(len(nodes)))
    if nodes != expected_nodes:
        raise ValueError(
            f"Test {test_id}: expected consecutive node IDs {expected_nodes[0]} to "
            f"{expected_nodes[-1]}, but received {nodes}."
        )

    if not np.allclose(distance.to_numpy(), distance.to_numpy().T):
        raise ValueError(f"{distance_path}: the distance matrix must be symmetric.")

    dummy_node = nodes[0]
    office_node = nodes[-1]
    employee_count = len(nodes) - 2
    expected_owner_index = pd.Index(nodes)
    car_owner = car_owner.reindex(expected_owner_index)
    if car_owner.isna().any():
        missing = car_owner[car_owner.isna()].index.tolist()
        raise ValueError(f"{car_owner_path}: missing values for nodes {missing}.")

    invalid_owner_values = sorted(set(car_owner.unique()) - {0.0, 1.0})
    if invalid_owner_values:
        raise ValueError(
            f"{car_owner_path}: car-owner values must be 0 or 1; found "
            f"{invalid_owner_values}."
        )

    return BaseInstance(
        test_id=test_id,
        employee_count=employee_count,
        dummy_node=dummy_node,
        office_node=office_node,
        car_owner=car_owner,
        schedule=schedule,
        distance_matrix=distance,
    )


def map_base_node(node: int, instance: BaseInstance) -> int:
    """Map a base node to its corresponding node in the V2 network."""

    if node == instance.dummy_node:
        return 0
    if node == instance.office_node:
        return 2 * instance.employee_count + 1
    return 2 * node - 1


def generate_v2_car_owner(instance: BaseInstance) -> pd.DataFrame:
    """Duplicate each employee's ownership flag across both pickup options."""

    new_size = 2 * instance.employee_count + 2
    values = pd.Series(0.0, index=range(new_size), dtype=float)
    values.loc[0] = float(instance.car_owner.loc[instance.dummy_node])

    for employee in range(1, instance.employee_count + 1):
        original_pickup = 2 * employee - 1
        alternative_pickup = 2 * employee
        owner_flag = float(instance.car_owner.loc[employee])
        values.loc[original_pickup] = owner_flag
        values.loc[alternative_pickup] = owner_flag

    new_office = new_size - 1
    values.loc[new_office] = float(instance.car_owner.loc[instance.office_node])
    return values.to_frame()


def generate_v2_distance_matrix(
    instance: BaseInstance, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Construct the deterministic V2 distance matrix for one test instance."""

    new_size = 2 * instance.employee_count + 2
    new_matrix = np.full((new_size, new_size), np.nan, dtype=float)

    # Preserve all distances between the dummy node, original employee nodes,
    # and the office node.
    for base_i in instance.distance_matrix.index:
        for base_j in instance.distance_matrix.columns:
            v2_i = map_base_node(int(base_i), instance)
            v2_j = map_base_node(int(base_j), instance)
            new_matrix[v2_i, v2_j] = float(
                instance.distance_matrix.loc[base_i, base_j]
            )

    # A fresh generator is used for every test.  This reproduces the published
    # data, which were generated independently with the same seed per instance.
    rng = random.Random(seed)
    for i in range(new_size):
        for j in range(i, new_size):
            if i == j or not np.isnan(new_matrix[i, j]):
                continue
            distance = round(rng.random() * 100, 2)
            new_matrix[i, j] = distance
            new_matrix[j, i] = distance

    np.fill_diagonal(new_matrix, 1000.0)

    # Prevent a route from visiting both representations of the same employee
    # consecutively.
    for employee in range(1, instance.employee_count + 1):
        original_pickup = 2 * employee - 1
        alternative_pickup = 2 * employee
        new_matrix[original_pickup, alternative_pickup] = 1000.0
        new_matrix[alternative_pickup, original_pickup] = 1000.0

    # The dummy start node has zero distance to and from every *other* node.
    # Its diagonal remains 1000, consistently with all other diagonal entries.
    new_matrix[0, 1:] = 0.0
    new_matrix[1:, 0] = 0.0

    if np.isnan(new_matrix).any():
        raise RuntimeError(
            f"Test {instance.test_id}: the generated matrix contains missing values."
        )
    if not np.allclose(new_matrix, new_matrix.T):
        raise RuntimeError(
            f"Test {instance.test_id}: the generated matrix is not symmetric."
        )

    node_ids = list(range(new_size))
    return pd.DataFrame(new_matrix, index=node_ids, columns=node_ids)


def write_v2_instance(
    instance: BaseInstance,
    output_dir: Path,
    seed: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Generate and write the two V2-specific files for one test."""

    car_owner_dir = output_dir / "Car_Owner"
    distance_dir = output_dir / "Distance_Matrix"
    car_owner_dir.mkdir(parents=True, exist_ok=True)
    distance_dir.mkdir(parents=True, exist_ok=True)

    car_owner_path = car_owner_dir / f"car_owner_test{instance.test_id}_v2.csv"
    distance_path = (
        distance_dir / f"distance_matrix_test{instance.test_id}_v2.csv"
    )

    existing = [path for path in (car_owner_path, distance_path) if path.exists()]
    if existing and not overwrite:
        files = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {files}. "
            "Use --overwrite to replace them."
        )

    generate_v2_car_owner(instance).to_csv(car_owner_path, header=False)
    generate_v2_distance_matrix(instance, seed=seed).to_csv(distance_path)
    return car_owner_path, distance_path


def generate_tests(
    input_dir: Path,
    output_dir: Path,
    test_ids: Iterable[int],
    seed: int,
    overwrite: bool,
) -> None:
    """Generate the selected V2 instances and print a concise summary."""

    generated_count = 0
    for test_id in test_ids:
        instance = load_base_instance(input_dir, test_id)
        car_owner_path, distance_path = write_v2_instance(
            instance=instance,
            output_dir=output_dir,
            seed=seed,
            overwrite=overwrite,
        )
        generated_count += 1
        print(
            f"Test {test_id:02d}: {instance.employee_count} employees -> "
            f"{2 * instance.employee_count} pickup nodes"
        )
        print(f"  car owner:      {car_owner_path}")
        print(f"  distance matrix: {distance_path}")

    print(
        f"Completed {generated_count} V2 test instance(s). "
        "Schedule files were not modified."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic car-owner and distance-matrix files for "
            "the alternative-pickup-point carpooling model (V2)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the base car_owner_testN.csv, "
            "schedule_testN.csv, and distance_matrix_testN.csv files. Both "
            "flat and Car_Owner/Schedule/Distance_Matrix layouts are supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Destination root. Car_Owner and Distance_Matrix subdirectories "
            "are created automatically. Default: the script directory."
        ),
    )
    parser.add_argument(
        "--tests",
        type=parse_test_ids,
        default=list(DEFAULT_TESTS),
        metavar="SPEC",
        help="Tests to generate, for example 1-24, 9,17, or 1-4,9,17.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed applied independently to each test (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing V2 output files.",
    )
    return parser


def main() -> None:
    """Run the command-line program."""

    args = build_argument_parser().parse_args()
    generate_tests(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        test_ids=args.tests,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
