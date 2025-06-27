import argparse
from args import args_parser
from server import FedProx


def main(args):
    fed_args = args_parser()
    fed_args.save_path = args.save_path
    fed_args.seed = args.seed
    fedProx = FedProx(fed_args)
    fedProx.server()
    fedProx.global_test()


if __name__ == '__main__':
    # random example
    random_seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save_path', type=str, default='exp1', help='path to save result file')
    args = parser.parse_args()
    for j in range(5):
        for i in range(10):
            args.seed = random_seed_list[i]
            args.save_path = f'50epoch40round/exp_{j + 1}'
            main(args)
