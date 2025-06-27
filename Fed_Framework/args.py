import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', type=int, default=50, help='number of rounds of training')
    parser.add_argument('--r', type=int, default=15, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=3, help='number of total clients')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--C', type=float, default=1, help='sampling rate')
    parser.add_argument('--B', type=int, default=32, help='local batch size')
    parser.add_argument('--mu', type=float, default=1, help='proximal term constant')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')

    parser.add_argument('--device', default=torch.device("cuda:4" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--gpu_id', type=int, default=4, help='gpu id to choose')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay per global round')
    parser.add_argument('--dataset', type=str, default='casme2', choices=['casme2', 'samm', 'casme3'])
    parser.add_argument('--num_classes', type=int, default=3, help='class number')
    parser.add_argument('--ratio', type=int, default=0.3, help='prior konwledge ratio')
    parser.add_argument('--save_path', type=str, default='exp1')
    parser.add_argument('--seed', type=int, default='42')
    clients = ["casme2", "samm", "casme3"]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args

