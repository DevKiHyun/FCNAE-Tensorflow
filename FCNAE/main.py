import argparse
import sys

sys.path.append(".")
import FCNAE.fcnae as fcnae
import FCNAE.train as train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--stride', type=int, default=2, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--training_epoch', type=int, default=200, help='-')
    parser.add_argument('--batch_size', type=int, default=128, help='-')
    parser.add_argument('--n_channel', type=int, default=1, help='-')
    args, unknown = parser.parse_known_args()

    FCNAE = fcnae.FCNAE(args)
    train.training(FCNAE, args)