from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse() -> Namespace:
    parser = ArgumentParser(prog="call_me_maybe")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/input/function_calling_tests.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/output/hello.json"))
    args: Namespace = parser.parse_args()
    print(args.functions_definition, args.input, args.output)
    return parser.parse_args()
