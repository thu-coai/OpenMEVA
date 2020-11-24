import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
np.random.seed(520)
from eva.utils import tokenize
from eva.union_metric.data.gen_train_data import build_vocab, build_data
import copy
import json
import os
import traceback
import sys

from logging import getLogger

logger = getLogger(__name__)

_CITATION = """\
@misc{guan2020union,
    title={UNION: An Unreferenced Metric for Evaluating Open-ended Story Generation},
    author={Jian Guan and Minlie Huang},
    year={2020},
    eprint={2009.07602},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
UNION, a learnable
UNreferenced metrIc for evaluating OpeneNded story generation, which measures the
quality of a generated story without any reference.

See the [README.md] file at https://github.com/thu-coai/UNION for more information.
"""

_KWARGS_DESCRIPTION = """
UNION score.

Args:

`predictions` (list of str): prediction/candidate sentences
`context` (list of str): context sentences
        
Returns:
    'scores': List of scores.
"""


class UNION():
    def __init__(self, tokenizer, output_dir, model_path, sen_split_token="<|endoftext|>"):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.sen_split_token = sen_split_token
        self.model_path = model_path

    def info(self):
        return {
            'description': _DESCRIPTION,
            'citation': _CITATION,
            'inputs description': _KWARGS_DESCRIPTION,
            'homepage': "https://github.com/thu-coai/UNION",
            'codebase urls': ["https://github.com/thu-coai/UNION"],
            'reference urls': ["https://github.com/thu-coai/UNION",
                               "https://arxiv.org/abs/2009.07602"]
        }

    def load_data(self, data):
        # np.random.shuffle(data)
        data_segnum = len(data) / 20.
        data_train = np.random.permutation(data[:int(data_segnum * 18)])
        data_dev = np.random.permutation(data[int(data_segnum * 18):int(data_segnum * 19)])
        data_test = np.random.permutation(data[int(data_segnum * 19):])
        print("data_train:", len(data_train), "data_dev:", len(data_dev), "data_test:", len(data_test))
        return data_train, data_dev, data_test

    def train(self, data, batch_size=10):
        data_train, data_dev, data_test = self.load_data(data)
        split_data = {"train": data_train, "dev": data_dev, "test": data_test}
        try:
            with open("%s/entity_vocab.json" % self.output_dir) as fin:
                vocab = json.load(fin)
            print("using cache file for vocab.")
        except:
            vocab = build_vocab(data_train, tokenizer=self.tokenizer, output_dir=self.output_dir)

        train_file_name, dev_file_name, test_file_name = "train", "dev", "test"
        flag = True
        for name in [train_file_name, dev_file_name, test_file_name]:
            for mode in ["positive", "negative"]:
                if not os.path.exists("%s/%s_%s.txt" % (self.output_dir, name, mode)):
                    flag = False
        if flag:
            print("using cache data for training")
        else:
            for name in [train_file_name, dev_file_name, test_file_name]:
                build_data(split_data[name], vocab=vocab, output_dir=self.output_dir, name=name,
                        tokenizer=self.tokenizer, sen_split_token=self.sen_split_token,
                        error_type_prob_dict = {"repeat":0.01, "replace":0.45, "shuffle":0.44, "negation":0.1},
                        error_time_prob_dict = {1:0.2, 2:0.3, 3:0.45, 4:0.05})
                        # error_type_prob_dict = {"repeat":0.1, "replace":0.3, "shuffle":0.4, "negation":0.2},
                        # error_time_prob_dict = {1:0.2, 2:0.3, 3:0.3, 4:0.2})
        path = os.path.split(os.path.realpath(__file__))[0]
        hparam = {"data_dir": self.output_dir,
                  "task_name": "train",
                  "output_dir": self.model_path,
                  "init_checkpoint": "%s/union_metric/model/uncased_L-12_H-768_A-12/bert_model.ckpt" % path,  # From https://github.com/google-research/bert
                  "bert_config_path": "%s/union_metric/model/uncased_L-12_H-768_A-12" % path,  # From https://github.com/google-research/bert
                  "train_file_name": "train",
                  "dev_file_name": "dev",
                  "test_file_name": "test",
                  "train_batch_size": batch_size,
                  "eval_batch_size": batch_size,
                  "predict_batch_size": batch_size,
                  }
        training_command = "python3 %s/union_metric/run_union.py" % (path)
        for param in hparam:
            training_command += " --%s %s" % (param, hparam[param])
        print("Execute command:", training_command)
        os.system(training_command)

    def compute(self, data, batch_size=32):
        """
        compute union score
        Args:
            data (list of dict including candidate and context):

        Returns:
             res (dict of list of scores): union score
        """
        pred_file_name = "pred"
        pro_data = [{"reference": tmp_data["candidate"], "context": tmp_data["context"]} for tmp_data in data]
        build_data(pro_data, only_positive=True, output_dir=self.output_dir, name=pred_file_name,
                   tokenizer=self.tokenizer, sen_split_token=self.sen_split_token)

        path = os.path.split(os.path.realpath(__file__))[0]

        ckpt_list = []
        for _, _, fnames in os.walk(self.model_path):
            for name in fnames:
                if name.endswith("meta"):
                    ckpt_list.append(".".join(name.split(".")[:2]))
        ckpt_list = sorted(list(set(ckpt_list)))

        # for ckpt in ckpt_list:
        #     path_name = "%s/%s_%s_%s.txt" % (self.model_path, pred_file_name, "probabilities", ckpt)
        #     if os.path.exists(path_name):
        #         os.system("rm %s" % (path_name))
        #         print("rm %s" % path_name)
        res = {}

        for ckpt in ckpt_list:
            # ckpt_num = int(ckpt.split("-")[-1])
            # if (ckpt_num not in [10000, 12000, 13000]) and ((ckpt_num <= 17000) or (ckpt_num >= 25000)):
            #     continue
            # print(ckpt)
            try:
                with open("%s/checkpoint" % self.model_path, "w") as fout:
                    fout.write('model_checkpoint_path: "%s"' % ckpt)
                hparam = {"data_dir": self.output_dir,
                        "task_name": "pred",
                        "output_dir": self.model_path,
                        "init_checkpoint": "%s/%s" % (self.model_path, ckpt),
                        "bert_config_path": "%s/union_metric/model/uncased_L-12_H-768_A-12" % path,
                        "test_file_name": pred_file_name,
                        "predict_batch_size": batch_size,
                        }
                training_command = "python3 %s/union_metric/run_union.py" % (path)
                for param in hparam:
                    training_command += " --%s %s" % (param, hparam[param])
                print("Execute command:", training_command)
                os.system(training_command)
                with open("%s/%s_%s_%s.txt" % (self.model_path, pred_file_name, "probabilities", ckpt)) as fin:
                    res["union-%s" % ckpt] = [float(line.strip()) for line in fin]
            except Exception as e:
                traceback.print_exc(e)
                continue
        return res
