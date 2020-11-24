# OpenEVA
安装：

```
cd ./OpenEVA
pip install -e ./
```

使用操作请看main.py

OpenEVA is an open-source and extensible toolkit that provides a unified framework to implement metrics for evaluating language generation especially for open-ended language generation (e.g., dialogue system, commonsense story generation, etc.). This package is designed for the following groups:

* **New to language generation**: We have hand-by-hand tutorials and detailed documents that can not only enable you to use relation extraction tools, but also help you better understand the research progress in this field.
* **Developers**: Our easy-to-use interface and high-performance implementation can acclerate your deployment in the real-world applications. Besides, we provide several pretrained models which can be put into production without any training.
* **Researchers**: With our modular design, various task settings and metric tools, you can easily carry out experiments on your own models with only minor modification. We have also provided several most-used benchmarks for different settings of relation extraction.
* **Anyone who need to submit an NLP homework to impress their professors**: With state-of-the-art models, our package can definitely help you stand out among your classmates!

This package is mainly contributed by 



## What is Language Generation Evaluation

 

## Toolkit Usage

### Union metric

1. Put your pre-trained model in eva/union_metric/model/
2. Prepare data
3. call `UNION.train(data)`
4. call `UNION.compute(data)`



## How to Cite

A good research work is always accompanied by a thorough and faithful reference. If you use or extend our work, please cite the following paper:

```
@inproceedings{han-etal-2019-opennre,
    title = "{O}pen{NRE}: An Open and Extensible Toolkit for Neural Relation Extraction",
    author = "Han, Xu and Gao, Tianyu and Yao, Yuan and Ye, Deming and Liu, Zhiyuan and Sun, Maosong",
    booktitle = "Proceedings of EMNLP-IJCNLP: System Demonstrations",
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-3029",
    doi = "10.18653/v1/D19-3029",
    pages = "169--174"
}
```

It's our honor to help you better explore relation extraction with our OpenNRE toolkit!

## Papers and Document

If you want to learn more about neural relation extraction, visit another project of ours ([NREPapers](https://github.com/thunlp/NREPapers)).

You can refer to our [document](https://opennre-docs.readthedocs.io/en/latest/) for more details about this project.

## Install 

### Install as A Python Package

We are now working on deploy OpenNRE as a Python package. Coming soon!

### Using Git Repository

Clone the repository from our github page (don't forget to star us!)

```bash
git clone https://github.com/thunlp/OpenNRE.git
```

If it is too slow, you can try

```
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

Then install all the requirements:

```
pip install -r requirements.txt
```

**Note**: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/. 

Then install the package with 

```
python setup.py install 
```

If you also want to modify the code, run this:

```
python setup.py develop
```

Note that we have excluded all data and pretrain files for fast deployment. You can manually download them by running scripts in the ``benchmark`` and ``pretrain`` folders. For example, if you want to download FewRel dataset, you can run

```bash
bash benchmark/download_fewrel.sh
```

## Easy Start

Make sure you have installed OpenNRE as instructed above. Then import our package and load pre-trained models.

```python
>>> import opennre
>>> model = opennre.get_model('wiki80_cnn_softmax')
```

Note that it may take a few minutes to download checkpoint and data for the first time. Then use `infer` to do sentence-level relation extraction

```python
>>> model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
('father', 0.5108704566955566)
```

You will get the relation result and its confidence score.

For now, we have the following available models:

* `wiki80_cnn_softmax`: trained on `wiki80` dataset with a CNN encoder.
* `wiki80_bert_softmax`: trained on `wiki80` dataset with a BERT encoder.
* `wiki80_bertentity_softmax`: trained on `wiki80` dataset with a BERT encoder (using entity representation concatenation).
* `tacred_bert_softmax`: trained on `TACRED` dataset with a BERT encoder.
* `tacred_bertentity_softmax`: trained on `TACRED` dataset with a BERT encoder (using entity representation concatenation).

## Training

You can train your own models on your own data with OpenNRE. In `example` folder we give example training codes for supervised RE models and bag-level RE models. You can either use our provided datasets or your own datasets.

## Google Group

If you want to receive our update news or take part in discussions, please join our [Google Group](