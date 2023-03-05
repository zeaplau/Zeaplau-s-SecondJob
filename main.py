import argparse
import re
import os
import torch
import logging
import random
import numpy as np
import torch.nn as nn

from datetime import datetime
from typing import Tuple
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.optimizer import Lion
from model.lstm_ref import LSTMRef
from model.load_convref import ConvRefDataset
from tools.tokenization import BasicTokenizer
from tools.create_config import ModelConfig
from tools.retrieve_kb import retrieve_ConvRef_KB, const_interaction_dic

ROOT_PATH = Path(os.getcwd())

# log
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)


def update_instance(instance):
    instance.current_paths = []
    instance.orig_candidate_paths = []
    instance.current_F1s = []
    instance.orig_F1s = []
    instance.F1s = []


def rank_cps(instance, dot_sim: torch.Tensor) -> Tuple[torch.Tensor, str]:
    score = torch.softmax(dot_sim, dim=2, dtype=torch.float)
    hit1_idx = torch.argmax(dot_sim, dim=2).item()
    path = instance.orig_candidate_paths[hit1_idx]
    hit1_entity = instance.path2ans[sum(path, ())]
    return score, hit1_entity


def get_loss_and_ans(model: LSTMRef, instance, cps_ids: torch.Tensor , time: int, alpha: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
    dot_sim, q_vec, ans_vecs, ga_vecs = model.forward(instance, cps_ids=cps_ids, time=time)
    rank_logits, hit1_entity = rank_cps(instance, dot_sim)
    loss = model.get_loss(dot_sim, instance.F1s, q_vec, ga_vecs, alpha)
    return rank_logits, hit1_entity, loss


def process(is_train, args, model: LSTMRef, optimizer: torch.optim.Optimizer, dataloader: DataLoader, kb_retriever) -> Tuple[int, float]:
    # retrieve kb and get path
    model.train() if is_train else model.eval()
    for step, instance in enumerate(dataloader):
        time = 0
        while time < len(instance['questions']):
            update_instance(instance=instance)
            cps, const_ans = retrieve_ConvRef_KB(instance=instance, kb_retriever=kb_retriever, tokenizer=tokenizer, time=time, is_train=is_train, not_update=True)
            if re.search("^%s" % const_interaction_dic, instance['question'][time]['question']):
                # TODO: calculate the const question directly use 'yes' or 'no'
                ...
            else:
                cps_ids = model.tokenize_sentence(cps)
                if is_train:
                    with torch.autograd.set_detect_anomaly(True):
                        rank_logits, hit1_entity, _loss = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)
                        optimizer.zero_grad()
                        _loss.backward()
                        optimizer.step()
                else:
                    with torch.no_grad():
                        rank_logits, hit1_entity, _ = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)
            # TODO Metrics

            # TODO Log
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset setting
    parser.add_argument("--dataset", default="ConvRef", type=str, help="Dataset for training.")
    parser.add_argument("--train_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_trainset.json", type=str, help="The path of trainset.")
    parser.add_argument("--dev_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_devset.json", type=str, help="The path of devset.")
    parser.add_argument("--test_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_testset.json", type=str, help="The path of testset.")
    parser.add_argument("--cache_dir", default=f"{ROOT_PATH}/ConvRef/", type=str, help="The cache for KB.")

    # Train setting
    parser.add_argument("--do_train", default=1, type=int, help="Train the model")
    parser.add_argument("--do_eval", default=1, type=int, help="Valid the model.")
    parser.add_argument("--do_test", default=0, type=int, help="Test the model.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for optimizer.")
    parser.add_argument("--vocab_txt", default=f"{ROOT_PATH}/config/vocab.txt", help="The vocabulary file for tokenizer.")
    parser.add_argument("--config", default=f"{ROOT_PATH}/config/config_SimpleRanker.json", type=str, help="The path of config file.")
    parser.add_argument("--alpha", default=0.5, type=int, help="The alpha value for loss calculate.")
    parser.add_argument("--batch_size", default=1, type=int, help="The batch size for model.")
    parser.add_argument("--seed", default=123, type=int, help="The seed for random generation.")
    parser.add_argument("--gpu_id", default=0, type=int, help="The gpu to use.")
    parser.add_argument("--output_dir", default=f"{ROOT_PATH}/result/", type=str, help="The dir for output.")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer for the model.")
    parser.add_argument("--checkpoint", default=datetime.now().strftime("%Y-%m-%d %H %M %S").replace(" ", "_"), type=str, help="Checkpoint name of save / load model.")

    # Model path
    parser.add_argument("--model_path", default=None, type=str, help="The path of the model.")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_id)
        logger.info(f"Cuda is available: {device}")
        args.gpu_id = 1
    else:
        device = "cpu"
        logger.info(f"Cuda is inavailable: {device}")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load dataset
    if args.do_train:
        train_set = ConvRefDataset(args.vocab_txt, "trainset")
        logger.info(f"----- Instances num: {len(train_set)} -----")
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    if args.do_eval:
        dev_set = ConvRefDataset(args.vocab_txt, "devset")
        logger.info(f"----- Instances num: {len(dev_set)} -----")
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
    if args.do_test:
        test_set = ConvRefDataset(args.vocab_txt, "testset")
        logger.info(f"----- Instances num: {len(test_set)} -----")
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    tokenizer = BasicTokenizer(args.vocab_txt)
    config = ModelConfig.from_json_file(args.config)
    model = LSTMRef(config=config, tokenizer=tokenizer, device=device)
    optimizer = AdamW(params=list(model.parameters()), lr=args.learning_rate) if args.optimzier == "AdamW" else Lion(param=list(model.parameters()), lr=args.learning_rate)
    
    if os.path.exists(args.checkpoint):
        try:
            model_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        except:
            assert f"{args.checkpoint} is not compatible for current model."
    ...
    # TODO process of train / valid / test

    # TODO save the model