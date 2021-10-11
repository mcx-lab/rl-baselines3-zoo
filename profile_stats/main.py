import pstats
from pstats import SortKey
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--stats-path", type=str)
parser.add_argument("-n", type=int, default=10)
args = parser.parse_args()
p = pstats.Stats(args.stats_path)
p.sort_stats(SortKey.TIME).print_stats(args.n)
