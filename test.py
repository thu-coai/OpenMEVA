import json
import os
import copy
import numpy as np
import sys

def test_metric(data):
    from eva.tokenizer import SimpleTokenizer, PretrainedTokenizer
    tokenizer = SimpleTokenizer(method="nltk")
    eva_path = "/home/guanjian/evaluation_dataset/metrics/OpenEva"

    metric_score = {}

    from eva.bleu import BLEU
    bleu_metric = BLEU(tokenizer=SimpleTokenizer(method="nltk"))
    # print(bleu_metric.info())
    metric_score.update(bleu_metric.compute(data))

    from eva.meteor import METEOR
    meteor_metric = METEOR()
    # print(meteor_metric.info())
    metric_score.update(meteor_metric.compute(data))

    from eva.Rouge import ROUGE
    rouge_metric = ROUGE()
    # print(rouge_metric.info())
    metric_score.update(rouge_metric.compute(data))

    from eva.fwppl import FwPPL
    fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="gpt2")
    # print(fwppl_metric.info())
    metric_score.update(fwppl_metric.compute(data))
    ft_fwppl_metric = FwPPL(model_id="gpt2", model_name_or_path="your_model_path")
    # print(ft_fwppl_metric.info())
    metric_score.update(ft_fwppl_metric.compute(data))

    from eva.bertscore import BERTScore
    bertscore_metric = BERTScore(model_type="bert-base-uncased")
    # print(bertscore_metric.info())
    metric_score.update(bertscore_metric.compute(data))

    from eva.bleurt import BLEURT # dependency: bleurt from: git+https://github.com/google-research/bleurt.git
    bleurt_metric = BLEURT(model_type="bleurt-tiny-512", model_path="your_model_path"%)
    # print(bleurt_metric.info())
    metric_score.update(bleurt_metric.compute(data))

    from eva.moverscore import MoverScore
    moverscore_metric = MoverScore()
    # print(moverscore_metric.info())
    metric_score.update(moverscore_metric.compute(data, batch_size=8))

    from eva.embed_metric import EMBED
    embedding_file = "/home/guanjian/glove/glove.6B.300d.txt" # replace the address with yours
    # you can change the function `pro_emb` in eva.utils.py to process the embedding file
    embed_metric = EMBED(embedding_file=embedding_file, tokenizer=tokenizer)
    # print(embed_metric.info())
    metric_score.update(embed_metric.compute(data))

    from eva.ruber import RUBER_refer, RUBER_unrefer, RUBER_refer_bert, RUBER_unrefer_bert, RUBER
    ruber_refer_metric = RUBER_refer(embedding_file=embedding_file, tokenizer=tokenizer)
    ruber_unrefer_metric = RUBER_unrefer(model_path="your_model_path")
    ruber_metric = RUBER(ruber_refer=ruber_refer_metric, ruber_unrefer=ruber_unrefer_metric)
    metric_score.update(ruber_metric.compute(data))

    ruber_refer_bert_metric = RUBER_refer_bert(model_id="bert-base-uncased")
    ruber_unrefer_bert_metric = RUBER_unrefer_bert(model_path="your_model_path", method="mean")
    ruber_bert_metric = RUBER(ruber_refer=ruber_refer_bert_metric, ruber_unrefer=ruber_unrefer_bert_metric, bert=True)
    metric_score.update(ruber_bert_metric.compute(data))

    from eva.union import UNION
    union_metric = UNION(tokenizer=tokenizer, output_dir="your_output_dir", model_path="your_model_path")
    # print(union_metric.info())
    metric_score.update(union_metric.compute(data, batch_size=2))

    for metric in metric_score:
        for d, r in zip(data, metric_score[metric]):
            d["metric_score"][metric] = r
    return data

def test_heva(data, figure_path="./figure"):
    from eva.heva import Heva
    heva = Heva([1,2,3,4,5])
    print("consistency of human evaluation:", heva.consistency([d["score"] for d in data]))
    
    # human evaluation, consistency & distribution
    data_model = {}
    for d in data:
        if d["model_name"] in data_model:
            data_model[d["model_name"]].append(np.mean(d["score"]))
        else:
            data_model[d["model_name"]] = [np.mean(d["score"])]

    print("="*20)
    for model_name in data_model:
        heva.save_distribution_figure(score=data_model[model_name], save_path=figure_path, model_name=model_name, ymin=0, ymax=50)
        for model2_name in data_model:
            print("mean score of %s: %.4f, %s: %.4f; " % (model_name, np.mean(data_model[model_name]), model2_name, np.mean(data_model[model2_name])), end="\t")
            mean_test_result = heva.mean_test(data_model[model_name], data_model[model2_name])
            print("(mean testing) t-statistic=%.4f, p-value=%.4f"%(mean_test_result["t-statistic"], mean_test_result["p-value"]))
    # correlation between human evaluation and automatic evaluation
    metric_name_list = list(data[0]["metric_score"].keys())
    data_metric = {}
    for d in data:
        for metric_name in metric_name_list:
            if metric_name in data_metric:
                data_metric[metric_name].append(d["metric_score"][metric_name])
            else:
                data_metric[metric_name] = [d["metric_score"][metric_name]]
    human_score = np.mean([d["score"] for d in data], 1)
    corr = {}
    for metric_name in metric_name_list:
        heva.save_correlation_figure(human_score, data_metric[metric_name], save_path=figure_path, metric_name=metric_name)
        corr[metric_name] = heva.correlation(data_metric[metric_name], human_score)
    print("="*20)
    print(corr)


if __name__ == "__main__":
    device = sys.argv[1] # "cuda:3", "cpu"

    if "cuda" in device:
        gpu_name = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
        print("using %s-th gpu"%gpu_name)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("using cpu")

    data = [
            {
                'context': "Jian is a student.",
                'reference': ["Jian comes from Tsinghua University. Jian is sleeping."],
                'candidate': "He comes from Beijing. He is sleeping.",
                'model_name': "human",
                'score': [5, 5, 5],
                'metric_score': {},
            },
            {
                'context': "Jian is a worker.",
                'reference': ["Jian came from China. Jian was running."],
                'candidate': "He came from China.",
                'model_name': "human",
                'score': [4, 4, 4],
                'metric_score': {},
            }
        ]

    figure_path="./figure"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    data_metric = test_metric(data)
    test_heva(data_metric)