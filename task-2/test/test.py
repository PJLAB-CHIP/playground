from argparse import ArgumentParser
from os import path as osp

from tritonPlayground.benchmark import BENCHMARK_REGISTRY


def main(args):
    for benchmark_name in args.benchmarks:
        benchmark = BENCHMARK_REGISTRY.get(benchmark_name)
        benchmark.run(
            save_path=osp.join(args.output_dir, benchmark_name), print_data=True
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Run benchmarks.")
    # e.g. python test.py --benchmarks add softmax
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        required=True,
        help="List of benchmarks to run.",
    )
    task_2_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    parser.add_argument(
        "--output_dir",
        type=str,
        default=osp.join(task_2_dir, "results"),
        help="Directory to save the results.",
    )
    args = parser.parse_args()

    main(args)
