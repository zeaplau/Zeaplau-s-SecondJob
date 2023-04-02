import os
import re
import datetime
import argparse
from typing import List

import pdb

def eval_hitk(logs: List[str], hitn: int) -> List[str]:
    def chunk(logs: List[str]):
        return [logs[i:i + 10] for i in range(0, len(logs), 10)]
    log_chunks = chunk(logs=logs)
    eval_result = {}
    hit_count = 0
    for c_idx, chunk in enumerate(log_chunks):
        topn_logs = chunk[:hitn]
        res = list((filter(lambda record: record[2] == record[3] or record[3] in record[2], topn_logs)))
        hit_count += 1 if len(res) != 0 else 0
        eval_result[c_idx] = res
    print(f"hit-{hitn} result: {hit_count}\nhit-{hitn} rate: {hit_count / len(eval_result.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="", type=str, help="Path of the logs file.")
    parser.add_argument("--hitn", default=10, type=int, help="hit@n result.")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        assert f"{args.log_path} is not exists"
    if args.hitn > 10:
        print("---- Only support not more than hit@10 -----")
        args.hitn = 10

    with open(args.log_path, "r", encoding="utf-8") as f:
        logs = f.readlines()
    logs = list(map(
        lambda x: [t.strip() for t in x], map(
        lambda x: x.split(","), map(
        lambda x: x.strip(), logs))))

    def convert_date(x):
        if re.search("\d+\-\d+\-\d+T\d+:\d+:\d+Z", x):
            return datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%d %B %Y")
        return x
    logs = list(map(lambda x: [convert_date(t) for t in x], logs))
    print(f"---- {args.log_path} ----")
    for hit in [1, 5, 10]:
        eval_hitk(logs=logs, hitn=hit)