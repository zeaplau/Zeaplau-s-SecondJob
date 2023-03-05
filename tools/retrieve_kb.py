import re
import copy
import numpy as np
from SPARQL_service import sparql_test
from datetime import datetime

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


def addin_historical_frontier(batch, kb_retriever, first_topic_entity, previous_topic_frontier, previous_ans_frontier, tokenizer):
    '''
        Maintain the entities appeared in the previous conversation.
    '''
    node_num = len(batch.historical_frontier) + len(first_topic_entity+previous_topic_frontier+previous_ans_frontier)
    for te in (first_topic_entity + previous_ans_frontier + previous_ans_frontier):
        if te not in batch.historical_frontier:
            batch.historical_frontier += [te]


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


def retrieve_ConvRef_KB(instance, kb_retriever, tokenizer, time, is_train, not_update):
    raw_candidate_paths, paths, instance.orig_F1s, topic_entities = [], {}, [], []

    # Retrieve the candidate answer paths
    if time == 0:
        topic_entity = instance['seed_entity']
        raw_candidate_paths = retrieve_via_frontier(frontier=topic_entity, topic_entity=topic_entity, raw_candidate_paths=raw_candidate_paths, kb_retriever=kb_retriever, question=instance['question'][time])
        instance.current_frontier = topic_entity
    else:
        topic_entity = instance['question'][time]['NER']
        if len(instance['question'][time]['gold_paths']) > 0:
            gold_p = instance['question'][time]['gold_paths']
            ans_frontier = list(set((filter(lambda x: re.search("^Q", x), map(lambda t: kb_retriever.wikidata_label_to_id(t), gold_p)))))
        else:
            prev_hit1p = sum([list(instance.path2ans[t])[:1] for t in instance.path2ans], [])
            ans_frontiers = list(map(lambda t: re.search("^Q", t), prev_hit1p))
            prev_frontiers = list(set(sum([[w for w in t if re.search('^Q', w)] for t in instance.path2ans], [])))
            sorted_frontiers = tuple(sorted(list(set(topic_entity + prev_frontiers))))
            topic_entity = list(filter(lambda x: x not in ['UNK'], map(lambda x: x if re.search("^Q", ) else kb_retriever.wikidata_label_to_id(x), sorted_frontiers)))
            instance.current_topics = topic_entity

            addin_historical_frontier(instance, kb_retriever, instance['seed_entity'], prev_frontiers, ans_frontiers)

            if re.search("^%s" % const_verification_dic, instance['questions'][time]['question']) and len(set(topic_entity)) > 0:
                frontier = set(topic_entity)
            else:
                frontier = set(topic_entity + instance.hitorical_frontier)
            
            raw_candidate_paths = retrieve_via_frontier(frontier, topic_entity, raw_candidate_paths, kb_retriever, instance['questions'][time]['question'], not_update=not_update)
    # TODO Score the candidata answer paths

    candidate_paths, hop_numbers = [], []
    max_cp_length, path2ans = 0, {}
    limit_number = 1000
    const_ans = 0

    for s_idx, statements in enumerate(raw_candidate_paths):
        filter_statements = {}
        for p_idx, p in enumerate(statements):
            if len(statements[p]) > 3:
                continue
            path2ans[sum(p, ())] = (statements[p], 0)
            filter_statements[p] = statements[p]
        instance.statements += [filter_statements]
    instance.path2ans = path2ans
    sorted_path = sorted(instance.path2ans.keys())
    ...
    gold_ans, do_month = clean_answer(instance['questions'][time]['gold_answer'])
    if re.search("^%s" % const_verification_dic, instance['questions'][time]['question']):
        gold_ans = [w for w in instance['questions'][time]['relation']] if instance['questions'][time]['relation'] != "" else instance.current_topics
        psesudo_ans = [w.lower() for w in instance['questions'][time]['gold_answer_text']]
    
    for p_idx, p in enumerate(sorted_path):
        pred_ans, _ = clean_answer(path2ans[p][0], do_month=do_month)
        if re.search("^%s" % const_verification_dic, instance['questions'][time]['question']) and is_train:
            measure_F1 = generate_Inclusion(gold_ans, set(p))
        elif re.search("^%s" % const_verification_dic, instance['questions'][time]['question']):
            measure_F1 = ['yes'] if generate_Inclusion(gold_ans, set(p)) == 1 else ['no']
            if len(set(topic_entity) == 0):
                measure_F1 = ['yes']
            measure_F1 = np.float16(measure_F1 == psesudo_ans)
        else:
            measure_F1 = generate_F1(pred_ans, gold_ans)
        
        p_txt = [kb_retriever.wikidata_id_to_label(w) for w in p if not re.search("^\?", w)]

        if (not is_train) or (np.random() < (limit_number * 1. / len(path2ans))) or measure_F1 > 0.5:
            if None in p_txt:
                continue
            
            path = []
            path = tokenizer.tokenize(" ".join(p_txt))
            path = tokenizer.convert_tokens_to_ids(path)
            try:
                pred_ans = tokenizer.tokenize(" ".join(pred_ans[:1]))
                pred_ans = tokenizer.convert_tokens_to_ids(pred_ans)
            except:
                pred_ans = [100]
            
            candidate_paths += [path]
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
        if re.search("^%s" % const_verification_dic, instance['questions'][time]['question']):
            if np.sum(instance.F1s) >= 1.:
                const_ans = gold_ans
        if np.sum(instance.F1s) == 0:
            instance.F1s[:] = 1.
        if np.sum(instance.F1s) == 0:
            instance.F1s = 1.
        
        return candidate_paths, const_ans