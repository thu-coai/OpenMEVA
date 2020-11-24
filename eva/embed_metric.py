import numpy as np
import re
from eva.ruber import RUBER_refer
_CITATION = """\
@inproceedings{rus2012optimal,
  title={An optimal assessment of natural language student input using word-to-word similarity metrics},
  author={Rus, Vasile and Lintean, Mihai},
  booktitle={International Conference on Intelligent Tutoring Systems},
  pages={675--676},
  year={2012},
  organization={Springer}
} for greedy matching

@article{landauer1997solution,
  title={A solution to Plato's problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge.},
  author={Landauer, Thomas K and Dumais, Susan T},
  journal={Psychological review},
  volume={104},
  number={2},
  pages={211},
  year={1997},
  publisher={American Psychological Association}
} for embedding average

@inproceedings{forgues2014bootstrapping,
  title={Bootstrapping dialog systems with word embeddings},
  author={Forgues, Gabriel and Pineau, Joelle and Larchev{\^e}que, Jean-Marie and Tremblay, R{\'e}al},
  booktitle={Nips, modern machine learning and natural language processing workshop},
  volume={2},
  year={2014}
} for vector extrema

@article{tao2017ruber,
  title={Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems},
  author={Tao, Chongyang and Mou, Lili and Zhao, Dongyan and Yan, Rui},
  journal={arXiv preprint arXiv:1701.03079},
  year={2017}
} for max_min_pooling
"""

_DESCRIPTION = """\
Embedding-based metric is an alternative to using word-overlap based metrics is to consider the meaning of each word as defined by a word embedding, which assigns a vector to each word.
Methods such as Word2Vec calculate these embeddings using distributional semantics; that is, they approximate the meaning of a word by considering how often it co-occurs with other words in the corpus.4 These embeddingbased metrics usually approximate sentence-level embeddings using some heuristic to combine the vectors of the individual words in the sentence.
The sentence-level embeddings between the candidate and target response are compared using a measure such as cosine distance.
"""

_KWARGS_DESCRIPTION = """
Embedding-based metric.

Args:
`embedding_file`: the name of glove embeddding file.
`data`: list of dict including reference and candidate.
`tokenizer`: a pre-defined tokenizer.

Returns:
    `res`: dict of list of scores.
"""

class EMBED():
    def __init__(self, embedding_file, tokenizer):
        self.ruber_refer = RUBER_refer(embedding_file, tokenizer, method=["greedy_matching", "vector_average", "vector_extrema"], name="embed_metric")

    def info(self):
        return {
            "description": _DESCRIPTION,
            "citation": _CITATION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data, scale_res=False):
        """
        compute embed metric using word2vec (referenced)
        Args:
            data (list of dict including reference and candidate):
            scale_res (bool): whether to scale result to (0,1)

        Returns:
             res (dict of list of scores): embed score
        """
        return self.ruber_refer.compute(data, scale_res)