"""Build the settings -> PSF -> image database.

    python scripts/build_database.py --n 300
    python scripts/build_database.py --n 1000 --dose 200     # with shot noise in the images

Output: output/database/sem_ray_db.npz (+ index.csv for a quick look in Excel).
Load it with semtwin.database.load_database().
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semtwin.column import SEMColumn
from semtwin.database import build_database

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300, help="number of random machine configurations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dose", type=float, default=None, help="counts per unit signal (adds Poisson noise)")
    p.add_argument("--out", default=str(ROOT / "output" / "database"))
    args = p.parse_args()
    build_database(SEMColumn(), n_configs=args.n, out_dir=args.out, seed=args.seed, dose=args.dose)


if __name__ == "__main__":
    main()
