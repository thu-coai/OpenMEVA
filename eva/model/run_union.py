import sys
if __name__ == "__main__":
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
    from eva.union import UNION
    union_metric = UNION(tokenizer=tokenizer, output_dir=sys.argv[1], model_path=sys.argv[2])
    union_metric.train(data, batch_size=2)
