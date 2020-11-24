#!/usr/bin/env python

import csv
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from eva.heva_utils.fleiss import fleissKappa
from eva.heva_utils.alpha import krippendorff_alpha
from eva.heva_utils.icc import icc
import kendall_w
import json
import matplotlib.pyplot as plt

class Heva():
    def __init__(self, avail_label_list):
        # avail_label_list = [1, 2, 3, 4, 5]
        self.avail_label_list = avail_label_list
        self.num_class = len(avail_label_list)
        self.score2label = {}
        for i, l in enumerate(sorted(avail_label_list)):
            self.score2label[l] = i

    def consistency(self, human_score_list):
        # human_score_list: [[1,2,1,2,3], [1,5,3,2,3], ...]
        if len(np.shape(human_score_list)) != 2:
            raise Exception("'human_score_list' must be a 2-dim array, where each line is a list of human scores for an example.")
        num_examples, num_raters = np.shape(human_score_list)
        score_matrix = []
        for i, score in enumerate(human_score_list):
            score_matrix.append([0 for _ in range(self.num_class)])
            for s in score:
                score_matrix[-1][self.score2label[s]] += 1
        score_matrix = np.array(score_matrix).tolist()
        icc_result = icc(human_score_list)
        return {
            "kappa": fleissKappa(score_matrix, num_raters),
            "icc & p-value": (icc_result[0], icc_result[4]),
            "kendall-w": kendall_w.compute_w(human_score_list),
            "krippendorff_alpha_nominal": krippendorff_alpha(human_score_list, "nominal"),
            "krippendorff_alpha_interval": krippendorff_alpha(human_score_list, "interval"),
            "krippendorff_alpha_ratio": krippendorff_alpha(human_score_list, "ratio"),
        }

    def save_distribution_figure(self, score, save_path, model_name, ymin=0, ymax=200):
        minvalue, maxvalue = min(self.avail_label_list), max(self.avail_label_list)
        bar_width = 0.25
        plt.cla()
        plt.hist(score, bins=int((maxvalue-minvalue)/bar_width), range=(minvalue, maxvalue))
        plt.xlabel("score of %s"%model_name)
        plt.ylabel("number")
        plt.ylim(ymax = ymax)
        plt.ylim(ymin = ymin)
        if not os.path.exists(save_path):
            os.system("mkdir %s"%save_path)        
        plt.savefig("%s/%s.pdf"%(save_path,model_name))

    def mean_test(self, model1_score, model2_score):
        # model1_score / model2_score: [1,2,3,4,...]
        mean_test_result = ttest_ind(model1_score, model2_score)
        return {
            "t-statistic": mean_test_result[0],
            "p-value": mean_test_result[1],
        }

    def save_correlation_figure(self, human_score, metric_score, save_path, metric_name):
        plt.cla()
        plt.plot(human_score, metric_score, ".")
        plt.xlabel("human")
        plt.ylabel(metric_name)
        if not os.path.exists(save_path):
            os.system("mkdir %s"%save_path)
        plt.savefig("%s/%s.pdf"%(save_path, metric_name))

    def correlation(self, human_score, metric_score):
        # human_score / metric_score: [1,2,3,4,...]
        return {
            "Pearson's Correlation": pearsonr(human_score, metric_score),
            "Spearman's Correlation": spearmanr(human_score, metric_score),
            "Kendall's Correlation": kendalltau(human_score, metric_score),
        }

    # issue_model = {}
    # for name in model_name:
    #     issue_model[name] = {}
    #     for iname in issue_name:
    #         issue_model[name][iname] = 0
    # for id_ in data:
    #     for name in model_name:
    #         for iname in data[id_]["gen"]["%s,topp0.9"%name]["score"]["issue"]:
    #             issue_model[name][iname] += 1
    # print(issue_model)