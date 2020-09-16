import argparse

def train(args):
    pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_val_embeddings', '-t', type=str, default=)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    train(args)