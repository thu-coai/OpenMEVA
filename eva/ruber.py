#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.3.28


'''
This file hybird the unreference score and reference score
'''
import numpy as np
from eva.utils import normalize_score, tokenize, pro_emb, word2vec, sent2vec, cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch
import tensorflow as tf
from eva.model.run_ruber_unrefer import ruber_unrefer_model, gen_batched_data, data2idx
from eva.model.run_ruber_unrefer_bert import ruber_unrefer_model_bert, gen_batched_data as gen_batched_data_bert
from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
import json
import os
import time
_CITATION = """\
@article{tao2017ruber,
  title={Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems},
  author={Tao, Chongyang and Mou, Lili and Zhao, Dongyan and Yan, Rui},
  journal={arXiv preprint arXiv:1701.03079},
  year={2017}
}
"""

_CITATION_BERT = """\
@inproceedings{ghazarian2019better,
  title={Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings},
  author={Ghazarian, Sarik and Wei, Johnny and Galstyan, Aram and Peng, Nanyun},
  booktitle={Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation},
  pages={82--89},
  year={2019}
}
"""

_DESCRIPTION = """\
RUBER is a Referenced metric and Unreferenced metric Blended Evaluation Routine, which evaluates a reply by taking into consideration both a groundtruth reply and a query (previous user-issued utterance). 
"""

_DESCRIPTION_BERT = """\
RUBER-BERT use contextualized word embeddings to compute more accurate relatedness scores, thus better evaluation metrics. 
"""

_KWARGS_DESCRIPTION = """
ruber score.

Args:

`embedding_file`: the name of glove embeddding file.
`data`: list of dict including reference and candidate.
`tokenizer`: a pre-defined tokenizer.
`model_path`: the path for trained model for unreferenced metric.
`batch_size`: batch size for computing the unreferenced metric.

Returns:
	`res`: dict of list of scores.
"""

_KWARGS_DESCRIPTION_BERT = """
ruber-bert score.

Args:

`model_id`: refer to `https://huggingface.co/models` for all available models.
`data`: list of dict including reference and candidate.
`model_path`: the path for trained model for unreferenced metric.
`batch_size`: batch size for computing the unreferenced metric.

Returns:
	`res`: dict of list of scores.
"""


class EmbedSimilar():
    def __init__(self):
        self.method = ["greedy_matching", "vector_average", "vector_extrema", "max_min"]

    def greedy_matching(self, cand_embed_list, ref_embed_list):
        all_cos_r = []
        for r in ref_embed_list:
            all_cos_r.append(max([cosine_similarity(r, c) for c in cand_embed_list]))
        sr = np.sum(all_cos_r) / len(all_cos_r)

        all_cos_c = []
        for c in cand_embed_list:
            all_cos_c.append(max([cosine_similarity(r, c) for r in ref_embed_list]))
        sc = np.sum(all_cos_c) / len(all_cos_c)
        return (sr + sc) / 2

    def vector_average(self, cand_embed_list, ref_embed_list):
        '''
        :embed_list: [embed1, embed2, embed3, ...]
        '''
        cand_average = list(np.sum(cand_embed_list, 0) / float(len(cand_embed_list)))
        ref_average = list(np.sum(ref_embed_list, 0) / float(len(ref_embed_list)))
        return cosine_similarity(cand_average, ref_average)

    def vector_extrema(self, cand_embed_list, ref_embed_list):
        def vector_extrema_helper(embed_list):
            sen_embed = []
            embed_dim = len(cand_embed_list[0])
            for i in range(embed_dim):
                maximum = max([e[i] for e in embed_list])
                minimum = min([e[i] for e in embed_list])
                if maximum > abs(minimum):
                    sen_embed.append(maximum)
                else:
                    sen_embed.append(minimum)
            return sen_embed

        cand_extrema = vector_extrema_helper(cand_embed_list)
        ref_extrema = vector_extrema_helper(ref_embed_list)
        return cosine_similarity(cand_extrema, ref_extrema)

    def max_min_pooling(self, cand_embed_list, ref_embed_list):
        def max_min_pooling_helper(embed_list):
            max_rest = np.max(embed_list, axis=0)
            min_rest = np.min(embed_list, axis=0)
            return np.concatenate([max_rest, min_rest])

        cand_pooling = max_min_pooling_helper(cand_embed_list)
        ref_pooling = max_min_pooling_helper(ref_embed_list)
        return cosine_similarity(cand_pooling, ref_pooling)

    def score(self, cand_embed_list, ref_embed_list, pooling_type="max_min"):
        if pooling_type == 'greedy_matching':
            pooling_sim = self.greedy_matching
        elif pooling_type == 'vector_average':
            pooling_sim = self.vector_average
        elif pooling_type == 'vector_extrema':
            pooling_sim = self.vector_extrema
        elif pooling_type == 'max_min':
            pooling_sim = self.max_min_pooling
        else:
            raise Exception(
                'pooling_type mush be one of ["greedy_matching", "vector_average", "vector_extrema", "max_min"]. Dafault is "max_min".')

        sim = pooling_sim(cand_embed_list, ref_embed_list)
        if np.isnan(sim):
            sim = 0.0
        return sim


class RUBER_refer(EmbedSimilar):
    def __init__(self, embedding_file, tokenizer, method=["max_min"], name="ruber_refer"):
        super(RUBER_refer, self).__init__()
        self.embed, _ = pro_emb(embedding_file)
        self.tokenizer = tokenizer
        self.method = method
        self.name = name
    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data, scale_res=True):
        res = {}
        for method in self.method:
            res["%s_%s" % (self.name, method)] = []
        for k, tmp_data in enumerate(data):
            st = time.time()
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            tmp_res = {}
            for method in self.method:
                tmp_res[method] = []
            for r in origin_reference:
                cand_embed_list = sent2vec(tokenize(origin_candidate, self.tokenizer), self.embed)
                ref_embed_list = sent2vec(tokenize(r, self.tokenizer), self.embed)
                for method in self.method:
                    tmp_res[method].append(
                        self.score(cand_embed_list=cand_embed_list, ref_embed_list=ref_embed_list, pooling_type=method))
            for method in self.method:
                res["%s_%s" % (self.name, method)].append(max(tmp_res[method]))
        if scale_res:
            for key in res:
                res[key] = normalize_score(res[key])
        return res


class RUBER_unrefer():
    def __init__(self, model_path, batch_size=32):
        self.model_path = model_path
        self.batch_size = batch_size
        with open("%s/hparam.json" % model_path) as fin:
            hparam = json.load(fin)
        class_name, method_name = tuple(hparam["tokenizer"].split("/"))
        self.tokenizer = SimpleTokenizer(method_name) if "SimpleTokenizer" in class_name else PretrainedTokenizer(
            method_name)
        self.max_sequence_length = hparam["max_sequence_length"]
        self.vocab_list = hparam["vocab_list"]
        self.vocab_size = hparam["vocab_size"]
        self.embed_dim = hparam["embed_dim"]
        self.hidden_dim = hparam["hidden_dim"]
        self.gen_func = gen_batched_data

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def pro_data(self, data):
        new_data = []
        for i, tmp_data in enumerate(data):
            new_data.append({})
            new_data[-1]["context"] = tokenize(tmp_data["context"], self.tokenizer,
                                               max_sequence_length=self.max_sequence_length)
            new_data[-1]["reference"] = tokenize(tmp_data["candidate"], self.tokenizer,
                                                 max_sequence_length=self.max_sequence_length)
        return data2idx(new_data, self.vocab_list)

    def score(self, model, sess, dataset):
        st, ed, prob = 0, 0, []
        while ed < len(dataset):
            st, ed = ed, ed + self.batch_size if ed + self.batch_size < len(dataset) else len(dataset)
            batch_data = self.gen_func(dataset[st:ed])
            outputs = model.step_decoder(sess, batch_data, forward_only=True)
            prob += list(outputs[2][:, 1])
        return prob

    def compute(self, data, scale_res=True):
        ruber_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        data = self.pro_data(data)
        with tf.Session(config=config, graph=ruber_graph) as sess:
            model = ruber_unrefer_model(
                vocab_size=self.vocab_size,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim)
            model.print_parameters()

            if tf.train.get_checkpoint_state(self.model_path):
                print("Reading model parameters from %s" % (self.model_path))
                model.saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
            if scale_res:
                score_list = normalize_score(self.score(model, sess, data))
            else:
                score_list = self.score(model, sess, data)

        return {"ruber_unrefer": score_list}


class RUBER_refer_bert(EmbedSimilar):
    def __init__(self, model_id):
        # refer to `https://huggingface.co/models` for all available models.
        super(RUBER_refer_bert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.method = ["max_min"]
    def info(self):
        return {
            "description for bert version": _DESCRIPTION_BERT,
            "citation for bert version": _CITATION_BERT,
            "inputs description for bert version": _KWARGS_DESCRIPTION_BERT,
        }

    def compute(self, data, scale_res=True):
        res = {}
        for method in self.method:
            res["ruber_refer_bert_%s" % method] = []
        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            tmp_res = {}
            for method in self.method:
                tmp_res[method] = []
            for r in origin_reference:
                cand_tokens = torch.tensor(self.tokenizer.encode(origin_candidate)).unsqueeze(0)
                ref_tokens = torch.tensor(self.tokenizer.encode(r)).unsqueeze(0)
                if torch.cuda.is_available():
                    cand_tokens = cand_tokens.cuda()
                    ref_tokens = ref_tokens.cuda()
                cand_embed_list = torch.squeeze(self.model(cand_tokens)[0]).detach().cpu().numpy().tolist()
                ref_embed_list = torch.squeeze(self.model(ref_tokens)[0]).detach().cpu().numpy().tolist()
                for method in self.method:
                    tmp_res[method].append(
                        self.score(cand_embed_list=cand_embed_list, ref_embed_list=ref_embed_list, pooling_type=method))

            for method in self.method:
                res["ruber_refer_bert_%s" % method].append(max(tmp_res[method]))
        if scale_res:
            for key in res:
                res[key] = normalize_score(res[key])
        return res


class RUBER_unrefer_bert():
    def __init__(self, model_path, method="max", batch_size=32):
        self.model_path = model_path
        self.batch_size = batch_size
        with open("%s/hparam.json" % model_path) as fin:
            hparam = json.load(fin)
        model_id = hparam["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.max_sequence_length = hparam["max_sequence_length"]
        self.gen_func = gen_batched_data_bert
        self.method = method

    def info(self):
        return {
            "description for bert version": _DESCRIPTION_BERT,
            "citation for bert version": _CITATION_BERT,
            "inputs description for bert version": _KWARGS_DESCRIPTION_BERT,
        }

    def pro_data(self, data, pooling_type):
        def pooling(sent):
            tokens = torch.tensor(self.tokenizer.encode(sent)[:self.max_sequence_length]).unsqueeze(0)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            embed = torch.squeeze(self.model(tokens)[0])
            if pooling_type == "max":
                embed_pooling = torch.max(embed, 0).values
            elif pooling_type == "mean":
                embed_pooling = torch.mean(embed, 0)
            else:
                raise Exception("pooling_type must be one of ['max', 'mean'].")
            return embed_pooling.detach().cpu().numpy().tolist()

        print("begin loading data ....")
        new_data = []
        for i, tmp_data in enumerate(data):
            new_data.append({})
            new_data[-1]["context"] = pooling(tmp_data["context"])
            new_data[-1]["reference"] = pooling(tmp_data["candidate"])
        return new_data

    def score(self, model, sess, dataset):
        st, ed, prob = 0, 0, []
        while ed < len(dataset):
            st, ed = ed, ed + self.batch_size if ed + self.batch_size < len(dataset) else len(dataset)
            batch_data = self.gen_func(dataset[st:ed])
            outputs = model.step_decoder(sess, batch_data, forward_only=True)
            prob += list(outputs[2][:, 1])
        return prob

    def compute(self, data, scale_res=True):
        ruber_bert_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        res = {}
        with tf.Session(config=config, graph=ruber_bert_graph) as sess:
            model = ruber_unrefer_model_bert(hidden_dim=self.model.config.hidden_size)
            model.print_parameters()

            if tf.train.get_checkpoint_state(self.model_path):
                print("Reading model parameters from %s" % (self.model_path))
                model.saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())
            data_ = self.pro_data(data, pooling_type=self.method)
            if scale_res:
                score_list = normalize_score(self.score(model, sess, data_))
            else:
                score_list = self.score(model, sess, data_)
            res["ruber_unrefer_bert_%s" % self.method] = score_list
        return res


class RUBER():
    def __init__(self, ruber_refer, ruber_unrefer, bert=False):
        self.ruber_refer = ruber_refer
        self.ruber_unrefer = ruber_unrefer
        self.method = ["min", "max", "geometric_mean", "arithmetic_mean"]
        self.bert = bert

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
            "description for bert version": _DESCRIPTION_BERT,
            "citation for bert version": _CITATION_BERT,
            "inputs description for bert version": _KWARGS_DESCRIPTION_BERT,
        }

    def hybird_score(self, refer_score_list, unrefer_list, method='min'):
        # make sure refer and unrefer has been normed
        if method == 'min':
            return [min(a, b) for a, b in zip(refer_score_list, unrefer_list)]
        elif method == 'max':
            return [max(a, b) for a, b in zip(refer_score_list, unrefer_list)]
        elif method == 'geometric_mean':  # geometric mean
            return [np.sqrt(a * b) for a, b in zip(refer_score_list, unrefer_list)]
        elif method == 'arithmetic_mean':  # arithmetic mean
            return [(a + b) / 2. for a, b in zip(refer_score_list, unrefer_list)]
        else:
            raise Exception('"method" must be one of ["min", "max", "geometric_mean", "arithmetic_mean"].')

    def compute(self, data):
        ruber_refer_score = self.ruber_refer.compute(data)
        ruber_unrefer_score = self.ruber_unrefer.compute(data)
        res = {}
        res.update(ruber_refer_score)
        res.update(ruber_unrefer_score)
        for method in self.method:
            for key1 in ruber_refer_score:
                for key2 in ruber_unrefer_score:
                    ruber_score = self.hybird_score(ruber_refer_score[key1], ruber_unrefer_score[key2], method=method)
                    res["%s_%s (with %s, %s)" % (
                        "ruber_bert" if self.bert else "ruber", method, key1, key2)] = normalize_score(ruber_score)
        return res
