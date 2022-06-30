import argparse

import trainer
from utils.logger import setup_logger


def main(args):
    if args.only_train_linear == 0:
        args.only_train_linear = False
    elif args.only_train_linear == 1:
        args.only_train_linear = True
    else:
        print('error')
        exit()
    print(args)
    setup_logger(args.output_dir)
    if args.only_eval:
        print('Do test ... ')
        trainer.test(args)
    else:
        print('Do train ... ')
        trainer.train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--net", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="aircraft")
    parser.add_argument("--use_pretrain_model", type=int, default=1)
    parser.add_argument("--only_train_linear", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--only_eval", type=bool, default=False)
    args = parser.parse_args()
    main(args)
