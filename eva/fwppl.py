from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

_DESCRIPTION = """\
FwPPL, or forward perplexity, is a metric which measure the language model score.
"""

_KWARGS_DESCRIPTION = """
FwPPL score.

Args:

`data`： (list of dict including reference and candidate).
`model_id`: refer to `https://huggingface.co/models` for all available models.
`model_name_or_path`: can be the same with model_id or a path of checkpoint.

Returns:
    `res`: dict of list of scores.
"""

class FwPPL():
    def __init__(self, model_id="gpt2", model_name_or_path="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        self.metric_name = "fwppl" if model_name_or_path==model_id else "ft-fwppl"


    def info(self):
        return {
            "description": _DESCRIPTION,
            "inputs description": _KWARGS_DESCRIPTION,
        }

    def compute(self, data):
        """
        compute fwppl score
        Args:
            data (list of dict including context and candidate):

        Returns:
             res (dict of list of scores): fwppl score
        """
        res = {self.metric_name: []}
        for tmp_data in data:
            origin_context = tmp_data['context'].strip() + " "
            origin_candidate = tmp_data['candidate'].strip()
            # "pt": pytorch; "tf": tensorflow
            context = self.tokenizer(origin_context, return_tensors='pt').input_ids
            candidate = self.tokenizer(origin_candidate, return_tensors='pt').input_ids
            whole_text = torch.cat([context, candidate], 1)

            input_ids = whole_text[:, :1000]
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            target_ids = input_ids.clone()
            target_ids[:,:len(context)] = -100
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                log_likelihood = outputs[0]

            res[self.metric_name].append(float(torch.exp(log_likelihood).cpu().numpy()))
        return res

if __name__ == "__main__":
    metric = FwPPL()
    print(metric.compute([
        {
            "context": "i have always hated the taste of almonds . they tried like crunchy forest . one day a friend bought me almonds covered in honey . i slowly placed one in my mouth .",
            "candidate": "i was joyful to find",
        },
        {
            "context": "i have always hated the taste of almonds . they tried like crunchy forest . one day a friend bought me almonds covered in honey . i slowly placed one in my mouth .",
            "candidate": "i was delightful to find",
        },
        {
            "context": "i have always hated the taste of almonds . they tried like crunchy forest . one day a friend bought me almonds covered in honey . i slowly placed one in my mouth .",
            "candidate": "i was happy to find",
        },        
        {
            "context": "i have always hated the taste of almonds . they tried like crunchy forest . one day a friend bought me almonds covered in honey . i slowly placed one in my mouth .",
            "candidate": "i was upset to find",
        },   
    ]
    )
    )