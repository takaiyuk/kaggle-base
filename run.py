import argparse
import os
import subprocess
import warnings

warnings.filterwarnings("ignore")


def run_fe():
    fe_file_list = [f for f in os.listdir("src/fe") if args.fe in f]
    assert fe_file_list
    fe_file: str = fe_file_list[0]
    cmd = f"python src/fe/{fe_file}"
    print(f"cmd: {cmd}")
    subprocess.run(cmd.split(" "))


def run_exp():
    exp_file_list = [e for e in os.listdir("src/exp") if args.exp in e]
    assert exp_file_list
    exp_file: str = exp_file_list[0]
    cmd = f"python src/exp/{exp_file}"
    print(f"cmd: {cmd}")
    subprocess.run(cmd.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_fe = subparsers.add_parser("fe", help="see `fe -h`")
    parser_fe.add_argument(
        "-f", "--fe", type=str, required=True, help="features config"
    )
    parser_fe.set_defaults(func=run_fe)
    parser_exp = subparsers.add_parser("exp", help="see `exp -h`")
    parser_exp.add_argument(
        "-e", "--exp", type=str, required=True, help="experiments config"
    )
    parser_exp.set_defaults(func=run_exp)
    args = parser.parse_args()
    args.func()
