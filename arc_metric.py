# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

import datasets
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score, classification_report


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ArcMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                "predictions": datasets.Value("int64"),
                "references": datasets.Value("int64"),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    #     # TODO: Download external resources if needed
    #     bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    #     self.bad_words = {w.strip() for w in open(bad_words_path, encoding="utf-8")}

    def _compute(self, predictions, references):
        """Returns the scores"""
        # acc = simple_accuracy(predictions, references)
        # average=None: return every class f1
        # f1_scores = f1_score(y_true=references, y_pred=predictions, average=None,zero_division=0)
        micro_f1 = f1_score(y_true=references, y_pred=predictions, average='micro',zero_division=0)
        # auc = roc_auc_score(references, predictions, multi_class='ovr')
        report = classification_report(y_true=references, y_pred=predictions,zero_division=0)
        print(report)
        return {
            # "accuracy": acc,
            # "macro_f1": f1_scores.mean(),
            "micro_f1": micro_f1,
            # 'classification_report':classification_report
            # "Inform_f1": f1_scores[0],
            # "Greeting_f1": f1_scores[1],
            # "Sighing_f1": f1_scores[2],
            # "Questioning_f1": f1_scores[3],
            # "Rhetorical_f1": f1_scores[4],
            # "Requring_f1": f1_scores[5],
            # "Begging_f1": f1_scores[6],
            # "Begging_f1": f1_scores[7],            
            # "Commissive_f1": f1_scores[8],
            # "Desiring_f1": f1_scores[9],
            # "Guessing_f1": f1_scores[10],
            # "Rogering_f1": f1_scores[11],            
            # "Agreeing_f1": f1_scores[12],
            # "Denying_f1": f1_scores[13],
            # "Condition_f1": f1_scores[14],
            # "Transition_f1": f1_scores[15],
        }
