import re
import copy
import json
import numpy as np

from datetime import datetime
from rank_bm25 import BM25Okapi

from tools.SPARQL_service import sparql_test
from tools.tokenization import BasicTokenizer

import pdb

const_minimax_dic = 'amount|number|how many|final|first|last|predominant|biggest|major|warmest|tallest|current|largest|most|newly|son|daughter'
const_interaction_dic = '(and|or)'
const_verification_dic = '(do|is|does|are|did|was|were)'

def clean_answer(raw_answer, do_month =False):
    a = list(raw_answer)[0]
    try:
        if re.search('T', a) and do_month: # for predict path
            return list(set([datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d") for a in raw_answer])), False
        elif re.search('T', a): # for predict path
            return list(set([datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y") for a in raw_answer])), False
        elif re.search('^\d+ \w+ \d{4}$', a):
            return list(set([datetime.strptime(a, "%d %B %Y").strftime("%Y-%m-%d") for a in raw_answer])), True
        elif re.search('^\d{4}$', a):
            return list(set([datetime.strptime(a, "%Y").strftime("%Y") for a in raw_answer])), False
    except:
        pass
    return list(raw_answer), False


def generate_Inclusion(pred_ans, gold_ans):
    return float(set(pred_ans).issubset(gold_ans))


def generate_F1(pred_ans, ans):
    TP = len(set(pred_ans) & set(ans))
    precision = TP*1./np.max([len(set(pred_ans)), 1e-10])
    recall = TP*1./np.max([len(set(ans)), 1e-10])
    F1 = 2. * precision * recall/np.max([(precision + recall), 1e-10])
    return F1


def addin_historical_frontier(batch, kb_retriever, first_topic_entity, previous_topic_frontier, previous_ans_frontier, time):
    '''
        Maintain the entities appeared in the previous conversation.
    '''
    # node_num = len(batch.historical_frontier) + len(first_topic_entity+previous_topic_frontier+previous_ans_frontier)

    if re.search("^%s" % const_verification_dic, batch.questions[time - 1]['question'].lower()):
        entities = set(first_topic_entity + previous_topic_frontier)
    else:
        entities = set(first_topic_entity + previous_ans_frontier + previous_topic_frontier)
    
    for te in entities:
        if te not in batch.historical_frontier and re.search("Q\d+", te):
            batch.historical_frontier += [te]
    batch.historical_frontier_text = list(
        map(lambda x: kb_retriever.wikidata_id_to_label(x), batch.historical_frontier)
    )


def retrieve_via_frontier(frontier, topic_entity, raw_candidate_paths, kb_retriever, question=None, do_debug=False, not_update=True):
    '''
    Retrieve candidate relation paths based on entities
    :param frontier: entities in the entity transition graph
    :param raw_candidate_paths: a collection of candidate relation paths
    :param question: question of this round
    :param do_debug: whether it is debug mode
    :param not_update: whether update cache or not
    :return: candidate relation paths for the current round of questions
    '''
    if do_debug: print('frontier ****', frontier)

    if len(topic_entity) == 2 and re.search(const_interaction_dic, question):
        topic_entity = tuple(sorted(topic_entity))
        const_type = tuple(re.findall('(?<= )%s(?= )' %const_interaction_dic, question))

        key = (topic_entity, const_type)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
        if const_type and (key not in kb_retriever.STATEMENTS):
            if not_update:
                statements = {}
            else:
                statements, sparql_txts = kb_retriever.SQL_1hop_interaction(((topic_entity[0],), (topic_entity[1],)), const_type)
                kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                kb_retriever.STATEMENTS[key].update(statements)
            #print('kb_retriever.STATEMENTS[(t, const_type)], mid', key, mytime.time()-time1)
        else:
            statements = kb_retriever.STATEMENTS[key]
            # print('cache kb_retriever.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
        if statements: raw_candidate_paths += [statements]
        #print('raw_candidate_paths', raw_candidate_paths); exit()

    for t in set(frontier):
        if not re.search('^Q', t): continue
        #print('kb_retriever.STATEMENTS[(t, None)]')
        key = (t, None)
        key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])

        if key not in kb_retriever.STATEMENTS:
            if not_update:
                statements = {}
            else:
                # print('t', t)
                statements, sparql_txts = kb_retriever.SQL_1hop(((t,),), kb_retriever.QUERY_TXT)
                #print('statements, sparql_txts', statements, sparql_txts)
                kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                statements_tmp, sparql_txts = kb_retriever.SQL_2hop(((t,),), kb_retriever.QUERY_TXT)
                kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                statements.update(statements_tmp)
                kb_retriever.STATEMENTS[key].update(statements)
                #print('kb_retriever.STATEMENTS[(t, None)]', key, mytime.time()-time1, statements)
        else:
            statements = kb_retriever.STATEMENTS[key]
            #print('cache kb_retriever.STATEMENTS[(t, None)]', mytime.time()-time1, statements)
        if statements: raw_candidate_paths += [statements]

        # If multiple entities involve in a question, other entities are treated as the constraints
        sorted_topic_entity = tuple(sorted(set(frontier) - set([t])))
        if len(sorted_topic_entity):
            key = (t, sorted_topic_entity)
            key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
            if key not in kb_retriever.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = kb_retriever.SQL_2hop_reverse(((t,),), set(frontier) - set([t]))
                    kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                    kb_retriever.STATEMENTS[key].update(statements)
                #print('kb_retriever.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            else:
                statements = kb_retriever.STATEMENTS[key]
                #print('cache kb_retriever.STATEMENTS[(t, sorted_topic_entity)]', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        # If min max keyword in the question, we treat it as the constraint
        if question is not None and re.search(const_minimax_dic, question):
            const_type = tuple(re.findall(const_minimax_dic, question))
            key = (t, const_type)
            key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
            if key not in kb_retriever.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = kb_retriever.SQL_1hop_reverse(((t,),), const_type)
                    kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                    kb_retriever.STATEMENTS[key].update(statements)
                #print('kb_retriever.STATEMENTS[(t, const_type)], mid', key, mytime.time()-time1)
            else:
                statements = kb_retriever.STATEMENTS[key]
                #print('cache kb_retriever.STATEMENTS[(t, const_type)], mid', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]

        #print('kb_retriever.STATEMENTS[(t, const_type)], year')
        # If the year string in the question, we should treat it as constraint
        if question is not None and re.search('[0-9][0-9][0-9][0-9]', question):
            const_type = tuple(re.findall('[0-9][0-9][0-9][0-9]', question))
            if (t, const_type) not in kb_retriever.STATEMENTS:
                if not_update:
                    statements = {}
                else:
                    statements, sparql_txts = kb_retriever.SQL_2hop_reverse(((t,),), const_type)
                    kb_retriever.QUERY_TXT = kb_retriever.QUERY_TXT.union(sparql_txts)
                    kb_retriever.STATEMENTS[(t, const_type)].update(statements)
                #print('kb_retriever.STATEMENTS[(t, const_type)], year', mytime.time()-time1)
            else:
                statements = kb_retriever.STATEMENTS[(t, const_type)]
                #print('cache kb_retriever.STATEMENTS[(t, const_type)], year', mytime.time()-time1)
            if statements: raw_candidate_paths += [statements]
    return raw_candidate_paths


def retrieve_ConvRef_KB(instance, kb_retriever: sparql_test, tokenizer: BasicTokenizer, time, is_train, not_update, oracle=0, q_tag: str="uncertain"):
    raw_candidate_paths, paths, instance.orig_F1s, topic_entities = [], {}, [], []

    # Retrieve the candidate answer paths
    if time == 0:
        # for ner in instance.questions[time]['NER']:
        #     ner_id = kb_retriever.wikidata_label_to_id(ner)
        #     if ner_id not in instance.historical_frontier and re.search("Q\d+", ner_id):
        #         instance.historical_frontier.append(ner_id)
        topic_entity = [instance.seed_entity]
        raw_candidate_paths = retrieve_via_frontier(frontier=topic_entity, topic_entity=topic_entity, raw_candidate_paths=raw_candidate_paths, kb_retriever=kb_retriever, question=instance.questions[time]['question'])
        instance.current_frontier = topic_entity
        instance.current_topics = topic_entity

        instance.historical_frontier = topic_entity
        instance.historical_frontier_text = list(set([instance.seed_entity_text]))
    else:
        prev_const = re.search("^%s" % const_verification_dic, instance.questions[time - 1]['question'].lower())
        topic_entity: str = instance.questions[time]['NER'] # topic entity in ner result

        # identify the ambiguous ner result using bm25 and neighbour entities
        if topic_entity != []:
            key = (instance.seed_entity, None)
            key = ' '.join([' '.join(list(r)) if isinstance(r, tuple) else str(r) for r in key])
            query_statements = kb_retriever.STATEMENTS.get(key, [])
            entities_in_hops = []
            for s in list(query_statements.values()):
                entities_in_hops += list(s)
            filter_entities = list(filter(lambda e: re.search("Q\d+", e), set(entities_in_hops)))

            # TODO spend too much time here

            entities_in_hops = list(map(lambda e: kb_retriever.wikidata_id_to_label(e, ner=True), filter_entities))
            entities_in_hops = list(filter(lambda x: x != 'UNK', entities_in_hops))
            corpus = list(
                map(lambda e: tokenizer.tokenize(e.lower()), entities_in_hops)
            )
            bm25 = BM25Okapi(corpus=corpus)
            querys = [tokenizer.tokenize(t.lower()) for t in topic_entity]
            entities_idx = [np.argmax(np.array(bm25.get_scores(query=q))).item() for q in querys]
            topic_entity = [entities_in_hops[idx] for idx in entities_idx]

        # FIXME
        prev_hit1p = sum([list(instance.path2ans[t])[:1] for t in instance.path2ans], []) if prev_const is None else {} # no answer entity if previous question is verification question
        ans_frontiers = list(filter(lambda t: re.search("^Q", t), instance.historical_frontier)) if oracle == 0 else list(filter(lambda x: re.search(x), re.search(instance.questions[time - 1]['gold_answer'])))
        prev_frontiers = list(set(sum([[w for w in t if re.search('^Q', w)] for t in instance.path2ans], [])))
        sorted_frontiers = sorted(list(set(topic_entity)))

        topic_entity = list(filter(lambda x: x not in ['UNK'], map(lambda x: x if re.search("^Q", x) else kb_retriever.wikidata_label_to_id(x), sorted_frontiers)))
        instance.current_topics = topic_entity

        # add the new entity and update historical frontier text
        addin_historical_frontier(instance, kb_retriever, instance.current_topics, prev_frontiers, ans_frontiers, time=time)

        if re.search("^%s" % const_verification_dic, instance.questions[time]['question'].lower()) and len(set(topic_entity)) > 0:
            frontier = set(topic_entity)
        else:
            frontier = set(topic_entity + instance.historical_frontier)

        raw_candidate_paths = retrieve_via_frontier(frontier, topic_entity, raw_candidate_paths, kb_retriever, instance.questions[time]['question'], not_update=not_update)

    # print("historical_frontier: {}".format(instance.historical_frontier))
    candidate_paths, filtered_cps_idx, hop_numbers, ts, p_entities = [], [], [], [], []
    max_cp_length, path2ans = 0, {}
    limit_number = 1000
    const_ans = 0

    for s_idx, statements in enumerate(raw_candidate_paths):
        filter_statements = {}
        for p_idx, p in enumerate(statements):
            if len(statements[p]) > 3:
                continue
            ts += [s_idx]
            path2ans[sum(p, ())] = (statements[p], 0)
            filter_statements[p] = statements[p]
        instance.statements += [filter_statements]
    instance.path2ans = path2ans
    sorted_path = sorted(instance.path2ans.keys())

    gold_ans, do_month = clean_answer([instance.questions[time]['gold_answer']])
    psesudo_ans = [w.lower() for w in [instance.questions[time]['gold_answer_text']]]

    # if re.search("^%s" % const_verification_dic, instance.questions[time]['question'].lower()):
    #     gold_ans = [w for w in instance.questions[time]['relation']] if instance.questions[time]['relation'] != "" else instance.current_topics
    #     psesudo_ans = [w.lower() for w in instance.questions[time]['gold_answer_text']]

    for p_idx, p in enumerate(sorted_path):
        # TODO filter paths according to the question type
        pred_ans, _ = clean_answer(path2ans[p][0], do_month=do_month)
        if re.search("^%s" % const_verification_dic, instance.questions[time]['question'].lower()) and is_train:
            measure_F1 = generate_Inclusion(gold_ans, set(p))
        elif re.search("^%s" % const_verification_dic, instance.questions[time]['question'].lower()):
            measure_F1 = ['yes'] if generate_Inclusion(gold_ans, set(p)) == 1 else ['no']
            if len(topic_entity) == 0:
                measure_F1 = ['yes']
            measure_F1 = np.float16(measure_F1 == psesudo_ans)
        else:
            measure_F1 = generate_F1(pred_ans, gold_ans)
        
        p_txt = [kb_retriever.wikidata_id_to_label(w) for w in p if not re.search("^\?", w)]
        # p_txt += list(
        #     map(lambda x: kb_retriever.wikidata_id_to_label(x) if re.search("Q\d+", x) else x, pred_ans)
        # )
        p_entities += list(map(lambda x: kb_retriever.wikidata_id_to_label(x) if re.search("Q\d+", x) else x, pred_ans))

        if q_tag == "time":
            # if predict entity is time
            try:
                if datetime.strptime(list(path2ans[p][0])[0], "%Y-%m-%dT%H:%M:%SZ") is not None:
                    filtered_cps_idx.append(p_idx)
            except:
                ...

        if (not is_train) or (np.random.rand() < (limit_number * 1. / len(path2ans))) or measure_F1 > 0.5:
            if None in p_txt:
                continue
            
            path = []
            # path = tokenizer.tokenize(" ".join(p_txt))
            # path = tokenizer.convert_tokens_to_ids(path)
            try:
                pred_ans = tokenizer.tokenize(" ".join(pred_ans[:1]))
                pred_ans = tokenizer.convert_tokens_to_ids(pred_ans)
            except:
                pred_ans = [100]
            
            candidate_paths += [p_txt]
            if p_txt not in instance.candidate_paths:
                instance.candidate_paths += [p_txt]
            if p not in instance.candidate_paths:
                instance.orig_candidate_paths += [p]

            F1 = measure_F1
            instance.current_F1s += [F1]
            instance.orig_F1s += [F1]
            instance.F1s += [F1]

            if len(path) > max_cp_length:
                max_cp_length = len(path)

    instance.F1s = np.array(instance.F1s)
    instance.current_F1s = np.array(instance.current_F1s)
    const_ans = None

    if len(filtered_cps_idx) != 0:
        candidate_paths = [candidate_paths[i] for i in filtered_cps_idx]
        instance.candidate_paths = [instance.candidate_paths[i] for i in filtered_cps_idx]
        instance.orig_candidate_paths = [instance.orig_candidate_paths[i] for i in filtered_cps_idx]
        instance.orig_F1s= [instance.orig_F1s[i] for i in filtered_cps_idx]
        instance.F1s = instance.F1s[filtered_cps_idx]
        instance.current_F1s = instance.current_F1s[filtered_cps_idx]

    if re.search("^%s" % const_verification_dic, instance.questions[time]['question'].lower()):
        if np.sum(instance.F1s) > 0.:
            const_ans = instance.questions[time]['gold_answer']
        else: # sum == 0
            const_ans = ['yes'] if instance.questions[time]['gold_answer'] == 'No' else 'No'

    if np.sum(instance.F1s) == 0:
        instance.F1s[:] = 1.
    if np.sum(instance.F1s) == 0:
        instance.F1s = 1.

    return candidate_paths, const_ans, ts