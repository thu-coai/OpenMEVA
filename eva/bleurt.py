import os
from logging import getLogger
from bleurt import score

logger = getLogger(__name__)

_CITATION = """\
@inproceedings{bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  booktitle={ACL},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}
"""

_DESCRIPTION = """\
BLEURT a learnt evaluation metric for Natural Language Generation. It is built using multiple phases of transfer learning starting from a pretrained BERT model (Devlin et al. 2018)
and then employing another pre-training phrase using synthetic data. Finally it is trained on WMT human annotations. You may run BLEURT out-of-the-box or fine-tune
it for your specific application (the latter is expected to perform better).

See the [README.md] file at https://github.com/google-research/bleurt for more information.
"""

_KWARGS_DESCRIPTION = """
BLEURT score.

Args:

`data`: list of dict including reference and candidate.
`model_type`: BLEURT checkpoint. Will default to bleurt-tiny-128 if None.
`add_context`: whether to add context to both the reference and candidate.

Returns:
    `res`: dict of list of scores.
"""

CHECKPOINT_URLS = {
    "bleurt-tiny-128": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip",
    "bleurt-tiny-512": "https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip",
    "bleurt-base-128": "https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip",
    "bleurt-base-512": "https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip",
    "bleurt-large-128": "https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip",
    "bleurt-large-512": "https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip",
}


class BLEURT():
    def __init__(self, model_type, model_path):
        self.config_name = model_type
        if self.config_name not in CHECKPOINT_URLS.keys():
            if self.config_name == 'default':
                logger.warning("Using default BLEURT-Base checkpoint for sequence maximum length 128. "
                               "You can use a bigger model for better results with e.g.: nlp.load_metric('bleurt', 'bleurt-large-512').")
                self.config_name = "bleurt-tiny-128"
            else:
                raise KeyError(
                    f"{self.config_name} model not found. You should supply the name of a model checkpoint for bleurt in {CHECKPOINT_URLS.keys()}")
        try:
            if model_path.endswith(self.config_name):
                self.scorer = score.BleurtScorer(model_path)
            else:
                self.scorer = score.BleurtScorer(os.path.join(model_path, self.config_name))
        except Exception as e:
            raise Exception(str(
                e) + f". You can download the checkpoint for {self.config_name} model from {CHECKPOINT_URLS[self.config_name]}")

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
            "homepage": "https://github.com/google-research/bleurt",
            "codebase urls": ["https://github.com/google-research/bleurt"],
            "reference urls": ["https://github.com/google-research/bleurt",
                               "https://arxiv.org/abs/2004.04696"],
            "checkpoint urls": CHECKPOINT_URLS,
        }

    def _compute(self, predictions, references):
        scores = self.scorer.score(references=references, candidates=predictions)
        return {"scores": scores}

    def compute(self, data, add_context=False):
        """
        compute bleurt score
        Args:
            data (list of dict including context, reference and candidate):

        Returns:
             res (dict of list of scores): bleurt score
        """
        res, map_idx, candidates, references = {}, [], [], []
        for i, tmp_data in enumerate(data):
            origin_context = tmp_data['context'].strip() + " " if add_context else ""
            origin_candidate = tmp_data['candidate'].strip()
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference.strip()]
            map_idx.append([])
            for r in origin_reference:
                candidates.append(origin_context + origin_candidate)
                references.append(origin_context + r)
                map_idx[i].append(len(candidates) - 1)
        bleurt_result = list(self._compute(predictions=candidates, references=references)["scores"])
        res["bleurt_score"] = [max([bleurt_result[i] for i in idx]) for idx in map_idx]
        return res
