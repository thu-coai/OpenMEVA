#!/usr/bin/env python

import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import json
import matplotlib.pyplot as plt
import copy
import os
import sys


def plot_fig(data):
    data_model = {"all":[]}
    for name in model_name:
        data_model[name] = []

    for d in data:
        name = d["model_name"]
        tmp_result = list(map(int, d["score"]))
        if len(tmp_result):
            data_model[name].append(np.mean(tmp_result))
            data_model["all"].append(np.mean(tmp_result))
    for name in model_name:
        print(name, np.mean(data_model[name]))
    # for name in model_name+["all"]:
    #     print(name, np.mean(data_model[name]))
    #     plt.cla()
    #     plt.hist(data_model[name], bins=16, range=(1,5))
    #     plt.ylim(ymin = 0)
    #     plt.ylim(ymax = 200)
    #     plt.savefig("./%s/%s.pdf"%(data_name, name))

    plt.cla()
    show_model = ["Seq2Seq", "Plan&Write", "Fusion", "GPT-2", "KG-GPT-2", "ALL"]
    show_model_map = {
        "Seq2Seq": "s2s",
        "Plan&Write": "plan_write",
        "Fusion": "fusion",
        "GPT-2":"gpt",
        "KG-GPT-2": "gpt_kg",
        "ALL": "all",
    }

    model_name+["all"]
    plt.figure(figsize=(6,3))
    bp = plt.boxplot([data_model[show_model_map[name]] for name in show_model], meanline=False, showmeans=True, meanprops={"markeredgecolor":"black", "markerfacecolor":"#99CCCC"}, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color="black")

    for patch, color in zip(bp["boxes"], ['#FFFFCC', '#CCFFFF', '#FFCCCC', '#FFCC99', '#CCFF99', '#CCCCCC']):
        patch.set(facecolor=color)

    plt.xticks([i for i in range(1, len(show_model)+1)], show_model)
    plt.ylabel("Annotation Score (%s)"%data_name.upper())

    plt.legend([bp["means"][0], bp["medians"][0]], ["Mean", "Median"], loc="best")
    plt.grid(True, axis="y")
    plt.savefig("./%s/box_%s.pdf"%(data_name, data_name))

def eva_human(data, heav):
    print("consistency of human evaluation:", heva.consistency([d["score"] for d in data]))
    # human evaluation, consistency & distribution
    data_model = {}
    for d in data:
        if d["model_name"] in data_model:
            data_model[d["model_name"]].append(np.mean(d["score"]))
        else:
            data_model[d["model_name"]] = [np.mean(d["score"])]
    dist_path = "./%s"%data_name
    with open("./%s/mean_test.txt"%dist_path, "w") as fout:
        for model_name in data_model:
            heva.save_distribution_figure(score=data_model[model_name], save_path=dist_path, model_name=model_name, ymin=0, ymax=50)
            for model2_name in data_model:
                fout.write("mean score of %s: %.4f, %s: %.4f; " % (model_name, np.mean(data_model[model_name]), model2_name, np.mean(data_model[model2_name])))
                mean_test_result = heva.mean_test(data_model[model_name], data_model[model2_name])
                fout.write("(mean testing) t-statistic=%.4f, p-value=%.4f\n"%(mean_test_result["t-statistic"], mean_test_result["p-value"]))

def stat(data):                
    import spacy
    from spacy.symbols import ORTH
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer.add_special_case("[MALE]", [{ORTH:"[MALE]"}])
    nlp.tokenizer.add_special_case("[FEMALE]", [{ORTH:"[FEMALE]"}])
    nlp.tokenizer.add_special_case("[NEUTRAL]", [{ORTH:"[NEUTRAL]"}])
    for w in [" WP ", " TT ", " IP ", " PM ", " EU ", " CW ", " FF ", " PI ", " OT ", " CS ", " RF ", " FF "]:
        nlp.tokenizer.add_special_case("[%s]"%w, [{ORTH:""}])
        nlp.tokenizer.add_special_case("[%s]"%w.lower(), [{ORTH:""}])

    num_context, num_reference, num_candidate = [], [], []
    for d in data:
        num_context.append(len(nlp(d["context"])))
        num_reference.append(len(nlp(d["reference"][0])))
        num_candidate.append(len(nlp(d["candidate"])))
    print("mean number for context:", np.mean(num_context))
    print("mean number for reference:", np.mean(num_reference))
    print("mean number for candidate:", np.mean(num_candidate))

def eval(data, heva):
    from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
    tokenizer = SimpleTokenizer(method="nltk")
    embedding_file = "/home/guanjian/glove/glove.6B.300d.txt"

    eva_path = "/home/guanjian/evaluation_dataset/metrics/OpenEva"

    metric_score = {}

    # from eva.bleu import BLEU
    # bleu_metric = BLEU(tokenizer=SimpleTokenizer(method="nltk"))
    # # print(bleu_metric.info())
    # metric_score.update(bleu_metric.compute(data))

    # from eva.meteor import METEOR
    # meteor_metric = METEOR()
    # # print(meteor_metric.info())
    # metric_score.update(meteor_metric.compute(data))

    # from eva.Rouge import ROUGE
    # rouge_metric = ROUGE()
    # # print(rouge_metric.info())
    # metric_score.update(rouge_metric.compute(data))

    from eva.fwppl import FwPPL
    # fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="gpt2")
    # # print(fwppl_metric.info())
    # metric_score.update(fwppl_metric.compute(data))
    tmp_data_name = "roc"
    ft_fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="%s/eva/model/%s_gpt2/checkpoint-%s00"%(eva_path, tmp_data_name, "275" if tmp_data_name=="roc" else "425"))
    metric_score.update(ft_fwppl_metric.compute(data))

    # from eva.bertscore import BERTScore
    # bertscore_metric = BERTScore(model_type="bert-base-uncased")
    # # print(bertscore_metric.info())
    # metric_score.update(bertscore_metric.compute(data))

    # from eva.bleurt import BLEURT # dependency: bleurt from: git+https://github.com/google-research/bleurt.git
    # bleurt_metric = BLEURT(model_type="bleurt-tiny-512", model_path="%s/eva/bleurt/model"%eva_path)
    # # print(bleurt_metric.info())
    # metric_score.update(bleurt_metric.compute(data))

    # from eva.moverscore import MoverScore
    # moverscore_metric = MoverScore()
    # # print(moverscore_metric.info())
    # metric_score.update(moverscore_metric.compute(data, batch_size=8))

    # from eva.embed_metric import EMBED
    # embed_metric = EMBED(embedding_file=embedding_file, tokenizer=tokenizer)
    # # print(embed_metric.info())
    # metric_score.update(embed_metric.compute(data))

    # ruber_model_path = "%s/eva/model/ruber_output_%s"%(eva_path, data_name)
    # ruber_model_path_bert = "%s/eva/model/ruber_output_bert{}_%s"%(eva_path, data_name)

    # from eva.model.run_ruber_unrefer import train_ruber_unrefer
    # train_ruber_unrefer(data, train_dir=ruber_model_path, embedding_file=embedding_file, tokenizer=tokenizer)
    # from eva.model.run_ruber_unrefer_bert import train_ruber_unrefer_bert
    # train_ruber_unrefer_bert(data, train_dir=ruber_model_path_bert, model_id="bert-base-uncased")

    # from eva.ruber import RUBER_refer, RUBER_unrefer, RUBER_refer_bert, RUBER_unrefer_bert, RUBER
    # ruber_refer_metric = RUBER_refer(embedding_file=embedding_file, tokenizer=tokenizer)
    # ruber_unrefer_metric = RUBER_unrefer(model_path=ruber_model_path)
    # ruber_metric = RUBER(ruber_refer=ruber_refer_metric, ruber_unrefer=ruber_unrefer_metric)
    # metric_score.update(ruber_metric.compute(data))

    # ruber_refer_bert_metric = RUBER_refer_bert(model_id="bert-base-uncased")
    # ruber_unrefer_bert_metric_max = RUBER_unrefer_bert(model_path=ruber_model_path_bert.format("_max"), method="max")
    # ruber_bert_metric_max = RUBER(ruber_refer=ruber_refer_bert_metric, ruber_unrefer=ruber_unrefer_bert_metric_max, bert=True)
    # metric_score.update(ruber_bert_metric_max.compute(data))
    # ruber_unrefer_bert_metric_mean = RUBER_unrefer_bert(model_path=ruber_model_path_bert.format("_mean"), method="mean")
    # metric_score.update(ruber_unrefer_bert_metric_mean.compute(data))
    # ruber_bert_metric_mean = RUBER(ruber_refer=ruber_refer_bert_metric, ruber_unrefer=ruber_unrefer_bert_metric_mean, bert=True)
    # metric_score.update(ruber_bert_metric_mean.compute(data))

    # from eva.union import UNION
    # # roc 9000, wp 4000
    # union_metric = UNION(tokenizer=tokenizer, output_dir="%s/eva/cache_%s"%(eva_path, data_name), model_path="%s/eva/union_metric/%s_output2"%(eva_path, data_name), sen_split_token=".")
    # metric_score.update(union_metric.compute(data, batch_size=10))


    for metric in metric_score:
        for d, r in zip(data, metric_score[metric]):
            d["metric_score"][metric] = r


    with open("./%s/%s_metric.json"%(data_name, data_name), "w") as fout:
        json.dump(data, fout, indent=4)

    # correlation between human evaluation and automatic evaluation
    corr_path = "./%s/corr_figure"%(data_name)
    metric_name_list = list(data[0]["metric_score"].keys())
    data_metric = {}
    for d in data:
        for metric_name in metric_name_list:
            if metric_name in data_metric:
                data_metric[metric_name].append(d["metric_score"][metric_name])
            else:
                data_metric[metric_name] = [d["metric_score"][metric_name]]
    human_score = np.mean([d["score"] for d in data], 1)
    corr = {}
    for metric_name in metric_name_list:
        heva.save_correlation_figure(human_score, data_metric[metric_name], save_path=corr_path, metric_name=metric_name)
        corr[metric_name] = heva.correlation(data_metric[metric_name], human_score)

    if os.path.exists("./%s/correlation.json"%data_name):
        with open("./%s/correlation.json"%data_name, "r") as fin:
            history_corr = json.load(fin)
            for c in history_corr:
                if c not in corr:
                    corr[c] = history_corr[c]

    with open("./%s/correlation.json"%data_name, "w") as fout:
        json.dump(corr, fout, indent=4)
    return data

device = "cuda:0"
# device = "cpu"

if "cuda" in device:
    gpu_name = device.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    print("using %s-th gpu"%gpu_name)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("using cpu")

#     for d in origin_data:
#         origin_data[d]['prompt'] = copy.deepcopy(origin_data[d]['beginning'])
#         del origin_data[d]['beginning']
# with open("%s_sample_1-200_ant_result_2.json"%data_name, "w") as fout:
#     json.dump(origin_data, fout, indent=4)
#     exit()

# data_name = "wp" # "wp"
data_name = sys.argv[1]
model_name = ["s2s", "fusion", "plan_write", "gpt", "gpt_kg"]
issue_name = ["rept", "rept3", "cohe", "cohe2", "chao", "conf", "conf2"]
with open("%s_sample_1-200_ant_result.json"%data_name, "r") as fin:
    origin_data = json.load(fin)
    data = []
    for id_ in origin_data:
        tmp_data = origin_data[id_]
        for name in model_name:
            d = {'context': tmp_data['context'] if 'context' in tmp_data else tmp_data['prompt'],
                'reference': [tmp_data['gold_response']],
                'candidate': tmp_data['gen'][name]["text"],
                'model_name': name,
                'id': id_}
            if "score" in tmp_data['gen'][name]:
                d["score"] = list(map(int, tmp_data['gen'][name]["score"]["quality_score"]))
            d["metric_score"] = {}
            data.append(d)

if not os.path.exists("./%s"%data_name):
    os.mkdir("./%s"%data_name)
if not os.path.exists("./%s/corr_figure"%(data_name)):
    os.mkdir("./%s/corr_figure"%(data_name))
from eva.heva import Heva
heva = Heva([1,2,3,4,5])

# plot_fig(data)
# eva_human(data, heva=heva)
# stat(data)
eval(data, heva)