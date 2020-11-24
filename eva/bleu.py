from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from eva.utils import tokenize


_CITATION = """\
@inproceedings{papineni2002bleu,
  title={BLEU: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of the 40th annual meeting of the Association for Computational Linguistics},
  pages={311--318},
  year={2002}
}
"""

_DESCRIPTION = """\
BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
"""

_KWARGS_DESCRIPTION = """
BLEU score.

Args:

`data`: (list of dict including reference and candidate).


Returns:
    `res`: dict of list of scores.
"""

class BLEU():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data):
        """
        compute bleu score
        Args:
            data (list of dict including reference and candidate): corpus data

        Returns:
            res (dict of list of scores): bleu score
        """
        res, candidates, references = {"sentence_bleu":[]}, [], []
        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            candidate = tokenize(origin_candidate, self.tokenizer)
            reference = [tokenize(r, self.tokenizer) for r in origin_reference]
            candidates.append(candidate)
            references.append(reference)
            res["sentence_bleu"].append(sentence_bleu(references=reference, hypothesis=candidate, smoothing_function=SmoothingFunction().method1))
        # res["corpus_bleu"] = corpus_bleu(list_of_references=references, hypotheses=candidates, smoothing_function=SmoothingFunction().method1)
        return res
