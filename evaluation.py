import evaluate
from transformers import EvalPrediction
import torch

class Evaluator:
    def __init__(self):
        self.exact_match = evaluate.load('exact_match')

    def evaluate(self, p: EvalPrediction):
        metrics_dict = dict()
        exact_match_metric = self.exact_match.compute(predictions=p.predictions, references=p.label_ids,
                                                      ignore_case=True, ignore_punctuation=True, regexes_to_ignore=' ')
        metrics_dict.update(exact_match_metric)
        return metrics_dict
