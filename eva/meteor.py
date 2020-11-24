import nltk
nltk.download('wordnet')
#need to download wordnet first

from nltk.translate.meteor_score import meteor_score as get_meteor_score

_CITATION = """\
@inproceedings{banerjee2005meteor,
  title={METEOR: An automatic metric for MT evaluation with improved correlation with human judgments},
  author={Banerjee, Satanjeev and Lavie, Alon},
  booktitle={Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization},
  pages={65--72},
  year={2005}
}
"""

_DESCRIPTION = """\
METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a metric for the evaluation of machine translation output.
The metric is based on the harmonic mean of unigram precision and recall, with recall weighted higher than precision.
It also has several features that are not found in other metrics, such as stemming and synonymy matching,
along with the standard exact word matching.
The metric was designed to fix some of the problems found in the more popular BLEU metric,
and also produce good correlation with human judgement at the sentence or segment level.
This differs from the BLEU metric in that BLEU seeks correlation at the corpus level.
"""

_KWARGS_DESCRIPTION = """
METEOR score.

Args:
`data`: (list of dict including reference and candidate).


Returns:
    `res`: dict of list of scores.
"""

class METEOR():
    def __init__(self, alpha=0.9, beta=3, gamma=0.5):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data):
        """
        compute meteor score
        Args:
            data (list of dict including reference and candidate):

        Returns:
             res (dict of list of scores): meteor score
        """
        res = {"meteor":[]}
        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            res["meteor"].append(get_meteor_score(references=origin_reference, hypothesis=origin_candidate, alpha=self.alpha, beta=self.beta, gamma=self.gamma))
        return res
