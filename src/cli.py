from argparse import ArgumentParser, Namespace
from pathlib import Path

def main():
    parser = ArgumentParser(prog="call_me_maybe")
    parser.add_argument("--functions_definition", type=Path)
    parser.add_argument("--input", type=Path, default=Path('data/input/'),)
    parser.add_argument("--output", type=Path, default=Path('data/output/'))
    args: Namespace = parser.parse_args()
    print(args.functions_definition, args.input, args.output)

if __name__ == '__main__':
    main()