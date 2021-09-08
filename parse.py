import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--train_batch', type=int, default=64)
    parser.add_argument('--val_batch', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='bing_co_order')
    parser.add_argument('--dataset_type', type=str, default='topk')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--randk', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--aggregator', type=str, default=None)
    parser.add_argument('--encoder', type=str, default='bert-base-uncased')
    parser.add_argument('--with_description', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=213214)
    parser.add_argument('--use_domain_emb', type=bool, default=False)

    return parser.parse_args()