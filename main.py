import argparse
import re
import os
import torch
import logging
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from typing import Tuple, List
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tools.SPARQL_service import sparql_test
from model.optimizer import Lion
from model.lstm_ref import RefModel
from model.load_convref import ConvRefDataset, ConvRefInstance
from tools.tokenization import BasicTokenizer
from tools.create_config import ModelConfig
from tools.retrieve_kb import retrieve_ConvRef_KB, const_interaction_dic

import pdb

ROOT_PATH = Path(os.getcwd())

# log setting
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)

# create kb_retriever
kb_retriever = sparql_test()


def generate_record(ins_idx: int, time: int, instance, score: torch.Tensor, gold_score: torch.Tensor, real_ans: str, topk: int=10, const_ans=None) -> List[str]:
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    if isinstance(gold_score, torch.Tensor):
        gold_score = gold_score.detach().cpu().numpy()
    
    # ins_idx time predict_ans real_ans is_recalled is_const
    path_format = "{} {} {} {} {} {}"
    logs = []
    new_p2a = {}

    if const_ans is None:
        topk_idx = list(map(lambda x: x[::-1][:topk], np.argsort(score, axis=1)))
        is_recalled = np.array(gold_score != 0.).any()
        # get predict path
        orig_paths = [instance.orig_candidate_paths[p_idx] for p_idx in topk_idx[0]] # we only use orignal question
        answer_entities = [list(instance.path2ans[p][0])[0] for p in orig_paths]
        
        new_cp = orig_paths[0]
        new_p2a[new_cp] = instance.path2ans[new_cp]

        # update path2ans
        if re.search("Q\d+", list(new_p2a[new_cp][0])[0]):
            instance.path2ans = new_p2a
        else:
            instance.path2ans = {}

        for e in answer_entities:
            logs += [path_format.format(ins_idx, time, e, real_ans, is_recalled, 0)]
    else:
        # we do not consider the recall here, since if we regard that if the system can not recall the message, 
        # give the answer as 'no' defaultly
        instance.path2ans = new_p2a
        is_recalled = 1. if (gold_score != 0.).any() else 0.
        for _ in range(topk):
            logs += [path_format.format(ins_idx, time, const_ans, real_ans, is_recalled, 1)]
    return logs


def update_instance(instance, const_ans):
    instance.current_paths = []
    instance.orig_candidate_paths = []
    instance.current_F1s = []
    instance.orig_F1s = []
    instance.F1s = []


def rank_cps(instance, dot_sim: torch.Tensor) -> Tuple[torch.Tensor, str]:
    new_p2a = {}
    score = F.softmax(dot_sim, dim=1)
    hit1_idx = torch.argmax(dot_sim, dim=1).detach().cpu().numpy()
    path = [instance.orig_candidate_paths[i] for i in hit1_idx]
    hit1_entity = [list(instance.path2ans[p][0])[0] for p in path]
    return score, hit1_entity


def get_loss_and_ans(model: RefModel, instance, cps_ids: torch.Tensor , time: int, alpha: int) -> Tuple[torch.Tensor, str, torch.Tensor]:
    entities_text = [kb_retriever.wikidata_id_to_label(x) for x in instance.historical_frontier]
    topic_entity, idx = model.choose_topic_entity(topic_entities=entities_text, question=instance.questions[time]['question'])
    ref_qs = model.rewrite(topic_entity=topic_entity, questions=[instance.questions[time]['question']] + instance.questions[time]['reformulations'])

    dot_sim, q_vec, ans_vecs, ga_vecs, nega_vecs = model.forward(instance, cps_ids=cps_ids, ref_qs=ref_qs, time=time)
    rank_logits, hit1_entity = rank_cps(instance, dot_sim)
    loss = model.get_loss(dot_sim, instance.F1s, q_vec, ga_vecs, nega_vecs, alpha)
    return rank_logits, hit1_entity, loss


def process(is_train, args, model: RefModel, optimizer: torch.optim.Optimizer, dataset, kb_retriever) -> Tuple[int, float, float, list]:
    """Process of train / valid / test
    """

    process_log_format = "avg_loss: {} avg_reward: {} reward_boundry: {}"

    # retrieve kb and get path
    total_loss, rewards, rewards_expect, path_logs = [], [], [], []
    hit1 = 0
    model.train() if is_train else model.eval()
    for step, instance in enumerate(dataset):
        time = 0
        const_ans = None
        instance.reset()
        while time < len(instance.questions):
            update_instance(instance=instance, const_ans=const_ans)

            const_ans = None
            cps, const_ans = retrieve_ConvRef_KB(instance=instance, kb_retriever=kb_retriever, tokenizer=tokenizer, time=time, is_train=is_train, not_update=True)
            if re.search("^%s" % const_interaction_dic, instance.questions[time]['question'].lower()):
                # TODO: calculate the const question directly use 'yes' or 'no'
                ...
            else:
                cps_join = [" ".join(cp) for cp in cps]
                cps_ids = model.tokenize_sentence(cps_join)
                if is_train:
                    with torch.autograd.set_detect_anomaly(True):
                        reward, reward_expect = 0., 0.
                        rank_logits, hit1_entity, _loss = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)

                        # if the question is const_verification question, we simply use the retrieve result to judge
                        # the question answer
                        if const_ans is None:
                            optimizer.zero_grad()
                            _loss.backward()
                            optimizer.step()
                            total_loss += [_loss.item()]
                            reward, reward_expect = torch.max(torch.softmax(rank_logits, dim=1)).detach().cpu().numpy().item(), np.max(instance.F1s).item()
                        else:
                            reward, reward_expect = 1. if instance.questions[time]['gold_answer'].lower() in const_ans else 0., 0.5

                        rewards += [reward]
                        rewards_expect += [reward_expect]

                        p_logs = generate_record(ins_idx=step, time=time, instance=instance, score=rank_logits, gold_score=instance.F1s, real_ans=instance.questions[time]['gold_answer'], topk=10, const_ans=const_ans)
                        path_logs += p_logs
                else:
                    with torch.no_grad():
                        if const_ans is None:
                            rank_logits, hit1_entity, _loss = get_loss_and_ans(model, instance, cps_ids, time, args.alpha)
                            total_loss += [_loss.item()]
                            reward, reward_expect = torch.max(torch.softmax(rank_logits, dim=2)).detach().cpu().numpy().item(), np.max(instance.F1s).item()
                        else:
                            reward, reward_expect = 1. if instance.questions[time]['gold_answer'].lower() in const_ans else 0., 0.5

                        rewards += [reward]
                        rewards_expect += [reward_expect]

                        p_logs = generate_record(ins_idx=step, time=time, instance=instance, score=rank_logits, gold_score=instance.F1s, real_ans=instance.questions[time]['gold_answer_entity'], topk=10, const_ans=const_ans)
                        path_logs += p_logs
            time += 1
        if is_train and step % 100 == 0:
            logger.info(f"avg_loss: {np.mean(total_loss)} avg_reward: {rewards}")

    avg_reward, avg_reward_boundry = np.mean(rewards), np.mean(rewards_expect)
    avg_loss = sum(total_loss) / (len(dataset) * 5)
    logger.info(process_log_format.format(avg_loss, avg_reward, avg_reward_boundry))
    return hit1, avg_loss, avg_reward, path_logs

def write_logs(path_logs: List[str], args):
    logger.info(f"Paths predict save at {ROOT_PATH}/ckpt/{args.checkpoint}")
    try:
        with open(f"{ROOT_PATH}/ckpt/{args.checkpoint}", "w", encoding="utf-8") as f:
            f.write("\n".join(path_logs))
    except:
        logger.info("Error happen.")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset setting
    parser.add_argument("--dataset", default="ConvRef", type=str, help="Dataset for training.")
    parser.add_argument("--train_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_trainset.json", type=str, help="The path of trainset.")
    parser.add_argument("--dev_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_devset.json", type=str, help="The path of devset.")
    parser.add_argument("--test_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_testset.json", type=str, help="The path of testset.")
    parser.add_argument("--debug_folder", default=f"{ROOT_PATH}/ConvRef/data/ConvRef_gold_debug.json", type=str, help="The path for debug.")

    # Train setting
    parser.add_argument("--do_train", default=1, type=int, help="Train the model")
    parser.add_argument("--do_eval", default=1, type=int, help="Valid the model.")
    parser.add_argument("--do_test", default=0, type=int, help="Test the model.")
    parser.add_argument("--do_debug", default=0, type=int, help="--debug")
    parser.add_argument("--epoch_nums", default=0, type=int, help="Epoch for training")
    parser.add_argument("--size", default=500, type=int, help="Train scale of each epoch.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate for optimizer.")
    parser.add_argument("--vocab_txt", default=f"{ROOT_PATH}/config/vocab.txt", help="The vocabulary file for tokenizer.")
    parser.add_argument("--config", default=f"{ROOT_PATH}/config/config_SimpleRanker.json", type=str, help="The path of config file.")
    parser.add_argument("--alpha", default=0.5, type=float, help="The alpha value for loss calculate.")
    parser.add_argument("--batch_size", default=1, type=int, help="The batch size for model.")
    parser.add_argument("--seed", default=123, type=int, help="The seed for random generation.")
    parser.add_argument("--gpu_id", default=0, type=int, help="The gpu to use.")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer for the model.")
    parser.add_argument("--checkpoint", default=datetime.now().strftime("%Y-%m-%d %H %M %S").replace(" ", "_"), type=str, help="Checkpoint name of save / load model.")
    parser.add_argument("--cache_dir", default=f"{ROOT_PATH}/ConvRef/cache/", type=str, help="The dir of cache.")

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

    cache_dir = args.cache_dir
    kb_retriever.load_cache(
        "%s/M2N.json" % cache_dir, 
        "%s/L2I.json" % cache_dir,
        "%s/STATEMENTS.json" % cache_dir, 
        "%s/QUERY.json" % cache_dir, 
        "%s/TYPE.json" % cache_dir, 
        "%s/OUTDEGREE.json" % cache_dir,
    )
    logger.info(f"Load cache from {cache_dir}")

    # Check cuda
    if torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_id)
        logger.info(f"Cuda is available, using: {device}")
        args.gpu_id = 1
    else:
        device = "cpu"
        logger.info(f"Cuda is inavailable, using: {device}")

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
    if args.do_debug:
        debug_set = ConvRefDataset(args.vocab_txt, "trainset", args.debug_folder).unique_convs
        logger.info(f"----- Instances num: {len(debug_set)} -----")
        debug_set = [ConvRefInstance(conv) for conv in debug_set]
    if args.do_train:
        train_set = ConvRefDataset(args.vocab_txt, "trainset", args.train_folder).unique_convs
        logger.info(f"----- Instances num: {len(train_set)} -----")
        train_set = [ConvRefInstance(conv) for conv in train_set]
        random.shuffle(train_set)
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if args.do_eval:
        dev_set = ConvRefDataset(args.vocab_txt, "devset", args.dev_folder).unique_convs
        logger.info(f"----- Instances num: {len(dev_set)} -----")
        dev_set = [ConvRefInstance(conv) for conv in dev_set]
        # dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    if args.do_test:
        test_set = ConvRefDataset(args.vocab_txt, "testset", args.test_folder).unique_convs
        logger.info(f"----- Instances num: {len(test_set)} -----")
        test_set = [ConvRefInstance(conv) for conv in test_set]
        # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    tokenizer = BasicTokenizer(args.vocab_txt)
    config = ModelConfig.from_json_file(args.config)
    model = RefModel(config=config, tokenizer=tokenizer, device=device)
    optimizer = AdamW(params=list(model.parameters()), lr=args.learning_rate) if args.optimizer == "AdamW" else Lion(param=list(model.parameters()), lr=args.learning_rate)
    
    if os.path.exists(args.checkpoint):
        try:
            model_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        except:
            assert f"{args.checkpoint} is not compatible for current model."

    ...
    init_hit, init_reward, init_loss = 0, 0., 99999.
    for epoch in range(args.epoch_nums):
        if args.do_debug:
            logger.info(f"Traninig epoch: {epoch}")
            hit, loss, reward, path_logs = process(is_train=1, args=args, model=model, optimizer=optimizer,dataset=debug_set, kb_retriever=kb_retriever)
            logger.info(f"Train epoch {epoch} hit {hit} avg_loss {loss} avg_reward {reward}")

        if args.do_train:
            logger.info(f"Traninig epoch: {epoch}")
            hit, loss, reward, path_logs = process(is_train=1, args=args, model=model, optimizer=optimizer,dataset=train_set[:args.size], kb_retriever=kb_retriever)
            logger.info(f"Train epoch {epoch} avg_loss {loss} avg_reward {reward}")
            random.shuffle(train_set)

        if args.do_eval:
            logger.info(f"Evaluating epoch: {epoch}")
            hit, loss, reward, path_logs = process(is_train=0, args=args, model=model, optimizer=optimizer, dataset=dev_set[:args.size / 5], kb_retriever=kb_retriever)
            logger.info(f"Valid epoch {epoch} hit {hit} avg_loss {loss} avg_reward {reward}")
            if reward > init_reward:
                init_reward = reward
                torch.save(model.state_dict(), f"{ROOT_PATH}/ckpt/valid/{args.checkpoint}.pth")
                logger.info(f"Model save at {ROOT_PATH}/ckpt/valid/{args.checkpoint}.pth")
                with open(f"{ROOT_PATH}/ckpt/valid/{args.checkpoint}.log", "w", encoding="utf-8") as f:
                    f.write("\n".join(path_logs))
                logger.info(f"Valid result save at {ROOT_PATH}/ckpt/{args.checkpoint}.log")
            kb_retriever.save_cache()

        if args.do_test:
            logger.info(f"Testing {ROOT_PATH}/ckpt/test/{args.checkpoint}.pth")
            hit, loss, reward, path_logs = process(is_train=0, args=args, model=model, optimizer=optimizer, dataset=test_set, kb_retriever=kb_retriever)
            with open(f"{ROOT_PATH}/ckpt/test/{args.checkpoint}.log", "w", encoding="utf-8") as f:
                f.write("\n".join(path_logs))
            logger.info(f"Test result save at {ROOT_PATH}/ckpt/test/{args.checkpoint}.logs")
            kb_retriever.save_cache()
    logger.info("End")