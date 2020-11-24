from moverscore_v2 import get_idf_dict, word_mover_score    # FROM git+https://github.com/AIPHES/emnlp19-moverscore.git
# from collections import defaultdict
import os

_CITATION = """\
@inproceedings{zhao2019moverscore,
  title={MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  author={Zhao, Wei and Peyrard, Maxime and Liu, Fei and Gao, Yang and Meyer, Christian M and Eger, Steffen},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={563--578},
  year={2019}
}
"""

_DESCRIPTION = """\
MoverScore is an metric combining contextualized representations with a distance measure perform the best.

See the [README.md] file at https://github.com/AIPHES/emnlp19-moverscore for more information.
"""

_KWARGS_DESCRIPTION = """
MoverScore score.

Args:

`data`: (list of dict including reference and candidate).


Returns:
    `res`: dict of list of scores.
"""

class MoverScore():
    def __init__(self, stop_words=[], n_gram=1):
        self.stop_words = stop_words
        self.n_gram = n_gram

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
            "homepage": "https://github.com/AIPHES/emnlp19-moverscore",
            "codebase urls": ["https://github.com/AIPHES/emnlp19-moverscore"],
            "reference urls": ["https://github.com/AIPHES/emnlp19-moverscore",
                            "https://arxiv.org/abs/1909.02622"],                        
        }

    def compute(self, data, batch_size=256):
        res, map_idx, candidates, references = {}, [], [], []
        for i, tmp_data in enumerate(data):
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            map_idx.append([])
            for r in origin_reference:
                candidates.append(origin_candidate)
                references.append(r)
                map_idx[i].append(len(candidates)-1)
        idf_dict_hyp = get_idf_dict(candidates) # idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = get_idf_dict(references) # idf_dict_ref = defaultdict(lambda: 1.)    
        moverscore_result = word_mover_score(references, candidates, idf_dict_ref, idf_dict_hyp, stop_words=self.stop_words, n_gram=self.n_gram, remove_subwords=True, batch_size=batch_size)
        res["moverscore"] = [max([moverscore_result[i] for i in idx]) for idx in map_idx]
        return res
