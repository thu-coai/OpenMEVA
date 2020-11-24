# import nlp
import bert_score

_CITATION = """\
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
"""

_DESCRIPTION = """\
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity.
It has been shown to correlate with human judgment on sentence-level and system-level evaluation. 
Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language generation tasks.

See the [README.md] file at https://github.com/Tiiiger/bert_score for more information.
"""

_KWARGS_DESCRIPTION = """
BERTScore Metrics with the hashcode from a source against one or more references.

Args:
    `predictions` (list of str): prediction/candidate sentences.
    `refereces` (list of str or list of list of str): reference sentences.
    `lang` (str): language of the sentences; required (e.g. 'en').
    `model_type` (str): bert specification, default using the suggested model for the target langauge; has to specify at least one of `model_type` or `lang`. see "https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0" for the list of supported models.
    `num_layers` (int): the layer of representation to use. default using the number of layer tuned on WMT16 correlation data
    `verbose` (bool): turn on intermediate status update.
    `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict.
    `device` (str): on which the contextual embedding model will be allocated on. If this argument is None, the model lives on cuda:0 if cuda is available.
    `nthreads` (int): number of threads.
    `batch_size` (int): bert score processing batch size. at least one of `model_type` or `lang`. `lang` needs to be. specified when `rescale_with_baseline` is True.
    `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline.
Returns:
    `res`: dict of list of scores, including 'precision', 'recall', 'f1'.
"""


class BERTScore():
    def __init__(
        self,
        lang=None,  # 'en' for roberta-large, 'en-sci' for scibert-scivocab-uncased, 'zh' for bert-base-chinese, 'others' for bert-base-multilingual-cased
        model_type=None,    # see "https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0" for the list of supported models
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        use_custom_baseline=False,
    ):
        if model_type is None:
            assert lang is not None, "either lang or model_type should be specified"
            model_type = bert_score.utils.lang2model[lang.lower()]

        if num_layers is None:
            num_layers = bert_score.utils.model2layers[model_type]

        hashcode = bert_score.utils.get_hash(model_type, num_layers, idf, rescale_with_baseline, use_custom_baseline)
        if not hasattr(self, 'cached_bertscorer') or self.cached_bertscorer.hash != hashcode:
            self.cached_bertscorer = bert_score.BERTScorer(
                model_type=model_type,
                num_layers=num_layers,
                batch_size=batch_size,
                nthreads=nthreads,
                all_layers=all_layers,
                idf=idf,
                device=device,
                lang=lang,
                rescale_with_baseline=rescale_with_baseline,
            )

        self.verbose, self.batch_size = verbose, batch_size

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "homepage": "https://github.com/Tiiiger/bert_score",
            "inputs description": _KWARGS_DESCRIPTION,
            "codebase urls": ["https://github.com/Tiiiger/bert_score"],
            "reference urls": ["https://github.com/Tiiiger/bert_score",
                            "https://arxiv.org/abs/1904.09675"],
        }

    def _compute(self, predictions, references):
        (P, R, F) = self.cached_bertscorer.score(
            cands=predictions, refs=references, verbose=self.verbose, batch_size=self.batch_size,
        )
        output_dict = {
            'precision': P,
            'recall': R,
            'f1': F,
        }
        return output_dict

    def compute(self, data):
        """
        compute bertscore
        Args:
            data (list of dict including reference and candidate):

        Returns:
             res (dict of list of scores): bertscore
        """
        res, candidates, references = {}, [], []
        for tmp_data in data:
            origin_candidate = tmp_data['candidate']
            origin_reference = tmp_data['reference']
            if not isinstance(origin_reference, list):
                origin_reference = [origin_reference]
            candidates.append(origin_candidate)
            references.append(origin_reference)

        result = self._compute(predictions=candidates, references=references)
        for key in result:
            res["bertscore-"+key] = result[key].numpy().tolist()
        return res
