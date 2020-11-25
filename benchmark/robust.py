#!/usr/bin/env python

import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import json
import matplotlib.pyplot as plt
import copy
import os
import sys


def eval(data, heva, metric_list, data_name, mode):
    metric_score = {}
    for metric in metric_list:
        metric_score.update(metric.compute(data))

    for metric in metric_score:
        for d, r in zip(data, metric_score[metric]):
            d["metric_score"][metric] = r

    with open("./%s_robust/%s_metric_%s.json"%(data_name, data_name, mode), "w") as fout:
        json.dump(data, fout, indent=4)

    corr_path = "./%s_robust/corr_figure"%(data_name)
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

    if os.path.exists("./%s_robust/correlation_%s.json"%(data_name, mode)):
        with open("./%s_robust/correlation_%s.json"%(data_name, mode), "r") as fin:
            history_corr = json.load(fin)
            for c in history_corr:
                if c not in corr:
                    corr[c] = history_corr[c]

    with open("./%s_robust/correlation_%s.json"%(data_name, mode), "w") as fout:
        json.dump(corr, fout, indent=4)
    return data
if __name__ == "__main__":
    for data_name in ["roc", "wp"]:
        for t1 in ["synonym_substitute", "semantic_substitute", "delete_punct", "contraction", "typos"]:
            for t2 in ["", "negative_sample_"]:
                mode = t2 + t1

                data = []
                if "negative" in mode:
                    with open("./data/acts_data/%s/invariance_test/%s_negative_sample.txt"%(data_name, data_name), "r") as fin:
                        negative_sample = [line.strip().split("|||")[2].strip() for line in fin]    
                    with open("./data/acts_data/%s/invariance_test/%s_%s_perturb.txt"%(data_name, data_name, mode), "r") as fin:
                        for line in fin:
                            tmp = line.strip().split("|||")
                            if len(tmp[2].strip()) == 0 or len(tmp[1].strip()) == 0:
                                continue
                            id_ = int(tmp[0])
                            data.append({
                                "context": tmp[1],
                                "candidate": tmp[2],
                                "metric_score": {},
                                "score": [0],
                            })
                            data.append({
                                "context": tmp[1],
                                "candidate": negative_sample[id_],
                                "metric_score": {},
                                "score": [1],
                            })
                else:
                    with open("./data/acts_data/%s_test_opt.txt"%data_name, "r") as fin:
                        truth = [line.strip() for line in fin]
                    with open("./data/acts_data/%s/invariance_test/%s_%s_perturb.txt"%(data_name, data_name, mode), "r") as fin:
                        for line in fin:
                            tmp = line.strip().split("|||")
                            if len(tmp[2].strip()) == 0 or len(tmp[1].strip()) == 0:
                                continue        
                            id_ = int(tmp[0])
                            data.append({
                                "context": tmp[1],
                                "candidate": tmp[2],
                                "metric_score": {},
                                "score": [0],
                            })
                            data.append({
                                "context": tmp[1],
                                "candidate": truth[id_].strip().lower() if "wp" in data_name else truth[id_].strip(),
                                "metric_score": {},
                                "score": [1],
                            })
                if not os.path.exists("./%s_robust"%data_name):
                    os.mkdir("./%s_robust"%data_name)
                if not os.path.exists("./%s_robust/corr_figure"%(data_name)):
                    os.mkdir("./%s_robust/corr_figure"%(data_name))
                from eva.heva import Heva
                heva = Heva([0,1])

                from eva.fwppl import FwPPL
                fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="gpt2")

                eval(data, heva, metric_list, data_name, mode)