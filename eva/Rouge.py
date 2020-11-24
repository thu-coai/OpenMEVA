from rouge import Rouge
import os

_CITATION = """\
@inproceedings{lin2004rouge,
  title={Rouge: A package for automatic evaluation of summaries},
  author={Lin, Chin-Yew},
  booktitle={Text summarization branches out},
  pages={74--81},
  year={2004}
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing.
"""

_KWARGS_DESCRIPTION = """
ROUGE score.

Args:

`data`： (list of dict including reference and candidate).


Returns:
    `res`: dict of list of scores.
"""

class ROUGE():
    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data):
        """
        compute rouge score
        Args:
            data (list of dict including reference and candidate):

        Returns:
             res (dict of list of scores): rouge score
        """
        rouge_name = ["rouge-1", "rouge-2", "rouge-l"]
        item_name = ["f", "p", "r"]

        rouge = Rouge()
        res = {}
        for name1 in rouge_name:
            for name2 in item_name:
                res["%s-%s"%(name1, name2)] = []
        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]

            tmp_res = []
            for r in origin_reference:
                tmp_res.append(Rouge().get_scores(refs=r, hyps=origin_candidate)[0])

            for name1 in rouge_name:
                for name2 in item_name:
                    res["%s-%s"%(name1, name2)].append(max([tr[name1][name2] for tr in tmp_res]))

        return res