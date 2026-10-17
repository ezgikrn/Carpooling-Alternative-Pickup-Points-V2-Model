# Carpooling Optimization with Alternative Pickup Points (V2)

This repository provides the input data and data-generation code for the V2 carpooling optimization model, in which each employee may be assigned to either their original home pickup point or an alternative pickup point.

## Repository Structure

- `Car_Owner/`: Car-ownership data for the 24 V2 test instances.
- `Distance_Matrix/`: Distance matrices for the 24 V2 test instances.
- `Schedule/`: Validated employee schedules used in the computational experiments.
- `pickup_point_generator_v2.py`: Python script used to generate the V2 car-owner and distance-matrix files from the corresponding V1 instances.

Each data directory contains files for Tests 1–24.

## Requirements

- Python 3.10 or later
- NumPy
- pandas

Install the required Python packages with:

```bash
pip install numpy pandas
```

## Input Data Required by the Generator

The generator requires the original V1 files for each test instance:

- `car_owner_testN.csv`
- `schedule_testN.csv`
- `distance_matrix_testN.csv`

The files may be stored in a single directory or in the following subdirectories:

- `Car_Owner/`
- `Schedule/`
- `Distance_Matrix/`

Here, `N` denotes the test number.

The corresponding V1 single-pickup-point model, source code, and base test instances are available in the [Carpooling Team repository](https://github.com/carpoolingteam/carpoolingteam). These V1 test instances can be used as the input data for the V2 generator provided in this repository.

## Usage

To generate all 24 V2 instances:

```bash
python pickup_point_generator_v2.py --input-dir ../testproblems --output-dir . --tests 1-24 --seed 123 --overwrite
```

To generate only selected instances, such as Tests 9 and 17:

```bash
python pickup_point_generator_v2.py --input-dir ../testproblems --output-dir . --tests 9,17 --seed 123 --overwrite
```

Use `--input-dir` to specify the directory containing the original V1 input files.

## Generated Files

The script generates:

- `Car_Owner/car_owner_testN_v2.csv`
- `Distance_Matrix/distance_matrix_testN_v2.csv`

The employee-level schedule files are not modified or regenerated. The validated schedules used in the computational experiments are provided separately in the `Schedule/` directory.

## Reproducibility

The alternative pickup-point data are generated deterministically using a user-specified random seed. The default seed is `123`.

Existing output files are replaced only when the `--overwrite` option is provided.
