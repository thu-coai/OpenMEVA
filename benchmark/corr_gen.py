#!/usr/bin/env python

import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import json
import matplotlib.pyplot as plt
import copy
import os
import sys

def eval(data, heva, metric_list):
    metric_score = {}
    for metric in metric_list:
        metric_score.update(metric.compute(data))

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

if __name__ == "__main__":
    for data_name in ["roc", "wp"]:
        # You can change model_name to test generalization across different generation models.
        model_name = ["s2s", "fusion", "plan_write", "gpt", "gpt_kg"]
        with open("./data/mags_data/mags_%s.json"%data_name, "r") as fin:
            origin_data = json.load(fin)
            data = []
            for id_ in origin_data:
                tmp_data = origin_data[id_]
                for name in model_name:
                    d = {'context': tmp_data['prompt'],
                        'reference': [tmp_data['gold_response']],
                        'candidate': tmp_data['gen'][name]["text"],
                        'model_name': name,
                        'id': id_}
                    if "score" in tmp_data['gen'][name]:
                        d["score"] = list(map(int, tmp_data['gen'][name]["score"]))
                    d["metric_score"] = {}
                    data.append(d)

        if not os.path.exists("./%s"%data_name):
            os.mkdir("./%s"%data_name)
        if not os.path.exists("./%s/corr_figure"%(data_name)):
            os.mkdir("./%s/corr_figure"%(data_name))

        from eva.heva import Heva
        heva = Heva([1,2,3,4,5])

        # Let's take BLEU as an example. You can try any other metric.
        # You can also test generalization across datasets by using learnable metrics trained on other datasets
        from eva.bleu import BLEU
        from eva.tokenizer import SimpleTokenizer
        bleu_metric = BLEU(tokenizer=SimpleTokenizer(method="nltk"))

        metric_data = eval(data, heva, [bleu_metric])