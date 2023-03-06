import argparse
import re
import os
import torch
import logging
import random
import numpy as np
import torch.nn as nn

from datetime import datetime
from typing import Tuple, List
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tools.SPARQL_service import sparql_test
from model.optimizer import Lion
from model.lstm_ref import LSTMRef
from model.load_convref import ConvRefDataset
from tools.tokenization import BasicTokenizer
from tools.create_config import ModelConfig
from tools.retrieve_kb import retrieve_ConvRef_KB, const_interaction_dic

ROOT_PATH = Path(os.getcwd())

# log setting
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)

# create kb_retriever
kb_retriever = sparql_test()

def generate_record(ins_idx: int, time: int, instance, score: torch.Tensor, gold_score: torch.Tensor, real_ans: str, topk: int=10, is_const: int=0) -> List[str]:
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    if isinstance(gold_score, torch.Tensor):
        gold_score = gold_score.detach().cpu().numpy()
    
    # ins_idx time predict_ans real_ans is_recalled is_const
    path_format = "{} {} {} {} {} {}"
    logs = []
    
    if not is_const:
        topk_idx = np.argmax(score, dim=2)[::-1][:topk]
        is_recalled = np.array(gold_score != 0.).any()
        # get predict path
        orig_paths = [instance.orig_candidate_paths[p_idx] for p_idx in topk_idx]
        answer_entities = [instance.path2ans[sum(p, ())] for p in orig_paths]
        for e in answer_entities:
            logs += [path_format.format(ins_idx, time, e, real_ans, is_recalled, is_const)]
    else:
        # we do not consider the recall here, since if we regard that if the system can not recall the message, 
        # give the answer as 'no' defaultly
        is_recalled = 'yes' if (gold_score != 0.).any() else 'no'
        e = 'yes' if (score != 0.).any() else 'no'
        for _ in range(topk):
            logs += [path_format.format(ins_idx, time, e, real_ans, is_recalled, is_const)]
    return logs


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


def process(is_train, args, model: LSTMRef, optimizer: torch.optim.Optimizer, dataloader: DataLoader, kb_retriever) -> Tuple[int, float, float, list]:
    """Process of train / valid / test
    """
    process_log_format = "avg_loss: {} hits:{} avg_reward: {} reward_boundry: {}"

    # retrieve kb and get path
    total_loss, rewards, rewards_expect, path_logs = [], [], [], []
    hit1 = 0
    model.train() if is_train else model.eval()
    for step, instance in enumerate(dataloader):
        time = 0
        while time < len(instance['questions']):
            update_instance(instance=instance)
            is_const = 0
            cps, const_ans = retrieve_ConvRef_KB(instance=instance, kb_retriever=kb_retriever, tokenizer=tokenizer, time=time, is_train=is_train, not_update=True)
            if re.search("^%s" % const_interaction_dic, instance['question'][time]['question']):
                # TODO: calculate the const question directly use 'yes' or 'no'
                is_const = 1
            else:
                cps_ids = model.tokenize_sentence(cps)
                if is_train:
                    with torch.autograd.set_detect_anomaly(True):
                        rank_logits, hit1_entity, _loss = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)
                        optimizer.zero_grad()
                        _loss.backward()
                        optimizer.step()
                        total_loss += [_loss.item()]
                        reward, reward_expect = torch.argmax(torch.softmax(rank_logits, dim=2)).detach().cpu().numpy().item(), np.max(instance.F1s).item()
                        rewards += [reward]
                        rewards_expect += [reward_expect]
                        
                        p_logs = generate_record(ins_idx=step, time=time, instance=instance, score=rank_logits, gold_score=instance.F1s, real_ans=instance['questions'][time]['gold_answer_entity'], topk=10, is_const=is_const)
                        path_logs += p_logs
                else:
                    with torch.no_grad():
                        rank_logits, hit1_entity, _loss = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)
                        total_loss += [_loss.item()]
                        reward, reward_expect = torch.argmax(torch.softmax(rank_logits, dim=2)).detach().cpu().numpy().item(), np.max(instance.F1s).item()
                        rewards += [reward]
                        rewards_expect += [reward_expect]

                        p_logs = generate_record(ins_idx=step, time=time, instance=instance, score=rank_logits, gold_score=instance.F1s, real_ans=instance['questions'][time]['gold_answer_entity'], topk=10, is_const=is_const)
                        path_logs += p_logs

    avg_reward, avg_reward_boundry = np.mean(rewards), np.mean(rewards_expect)
    avg_loss = sum(total_loss) / (len(dataloader) * 5)
    logger.info(process_log_format.format(avg_loss, avg_reward, avg_reward_boundry))
    return hit1, avg_loss, avg_reward, path_logs

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
    parser.add_argument("--epoch_nums", default=0, type=int, help="Epoch for training")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for optimizer.")
    parser.add_argument("--vocab_txt", default=f"{ROOT_PATH}/config/vocab.txt", help="The vocabulary file for tokenizer.")
    parser.add_argument("--config", default=f"{ROOT_PATH}/config/config_SimpleRanker.json", type=str, help="The path of config file.")
    parser.add_argument("--alpha", default=0.5, type=int, help="The alpha value for loss calculate.")
    parser.add_argument("--batch_size", default=1, type=int, help="The batch size for model.")
    parser.add_argument("--seed", default=123, type=int, help="The seed for random generation.")
    parser.add_argument("--gpu_id", default=0, type=int, help="The gpu to use.")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer for the model.")
    parser.add_argument("--checkpoint", default=datetime.now().strftime("%Y-%m-%d %H %M %S").replace(" ", "_"), type=str, help="Checkpoint name of save / load model.")

    # Model path
    parser.add_argument("--model_path", default=None, type=str, help="The path of the model.")

    args = parser.parse_args()

    # Log
    os.makedirs("logs", exist_ok=True)
    f_formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    f_handler = logging.FileHandler(f"{ROOT_PATH}/logs/{args.checkpoint}.log")
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)
    logger.info(f"Log file at {ROOT_PATH}/logs/{args.checkpoint}.log")

    # Chech cuda
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

    os.makedirs(f"{ROOT_PATH}/ckpt/", exist_ok=True)
    os.makedirs(f"{ROOT_PATH}/ckpt/trainn", exist_ok=True)
    os.makedirs(f"{ROOT_PATH}/ckpt/valid/", exist_ok=True)
    os.makedirs(f"{ROOT_PATH}/ckpt/test/", exist_ok=True)

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
    init_hit, init_reward, init_loss = 0, 9999
    for epoch in args.epoch_nums:
        if args.do_train:
            logger.info(f"Traninig epoch: {epoch}")
            hit, loss, reward, path_logs = process(is_train=1, args=args, model=model, optimizer=optimizer, dataloader=train_loader, kb_retriever=kb_retriever)
            logger.info(f"Train epoch {epoch} hit {hit} avg_loss {loss} avg_reward {reward}")

        if args.do_eval:
            logger.info(f"Evaluating epoch: {epoch}")
            hit, loss, reward, path_logs = process(is_train=0, args=args, model=model, optimizer=optimizer, dataloader=dev_loader, kb_retriever=kb_retriever)
            logger.info(f"Valid epoch {epoch} hit {hit} avg_loss {loss} avg_reward {reward}")
            if reward > init_reward:
                init_reward = reward
                torch.save(model.state_dict(), f"{ROOT_PATH}/ckpt/{args.checkpoint}.pth")
                logger.info(f"Model save at {ROOT_PATH}/ckpt/{args.checkpoint}.pth")
                with open(f"{ROOT_PATH}/ckpt/valid/{args.checkpoint}.log", "w", encoding="utf-8") as f:
                    f.write("\n".join(path_logs))
                logger.info(f"Valid result save at {ROOT_PATH}/ckpt/{args.checkpoint}.log")
            kb_retriever.save_cache()

        if args.do_test:
            logger.info(f"Testing {ROOT_PATH}/ckpt/{args.checkpoint}.pth")
            hit, loss, reward, path_logs = process(is_train=0, args=args, model=model, optimizer=optimizer, dataloader=test_loader, kb_retriever=kb_retriever)
            with open(f"{ROOT_PATH}/ckpt/test/{args.checkpoint}.log", "w", encoding="utf-8") as f:
                f.write("\n".join(path_logs))
            logger.info(f"Test result save at {ROOT_PATH}/ckpt/{args.checkpoint}.logs")
            kb_retriever.save_cache()