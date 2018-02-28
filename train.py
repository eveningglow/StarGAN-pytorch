import torch
import argparse
import os

import model
from solver import Solver

def main(args):
    solver = Solver(img_dir = args.img_dir,
                    ann_path = args.ann_path,
                    result_dir = args.result_dir,
                    weight_dir = args.weight_dir,
                    batch_size = args.batch_size,
                    lr = args.lr,
                    beta1 = args.beta1,
                    beta2 = args.beta2,
                    lambda_gp = args.lambda_gp,
                    lambda_recon = args.lambda_recon,
                    n_critic = args.n_critic,
                    class_num = args.class_num,
                    num_epoch = args.num_epoch,
                    save_every = args.save_every,
                    load_weight = args.load_weight)
                    
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/img')
    parser.add_argument('--ann_path', type=str, default='data/list_attr_celeba.txt')
    parser.add_argument('--result_dir', type=str, default='result')
    parser.add_argument('--weight_dir', type=str, default='weight')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_recon', type=float, default=10)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--load_weight', action='store_true')

    args = parser.parse_args()
    
    main(args)
