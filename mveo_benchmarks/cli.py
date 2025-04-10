# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys


def main():
    """Console script for mveo_benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("_", nargs="*")
    args = parser.parse_args()

    print("Arguments: " + str(args._))
    print("Replace this message by putting your code into " "mveo_benchmarks.cli.main")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
