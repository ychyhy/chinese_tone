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
class ConfusionMatrixMetric(datasets.Metric):
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


    def _compute(self, predictions, references):
        """Returns the scores"""
        micro_f1 = f1_score(y_true=references, y_pred=predictions, average='micro',zero_division=0)
        report = classification_report(y_true=references, y_pred=predictions,zero_division=0)
        print(report)
        # 测试集混淆矩阵
        y_pred = predictions
        #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
        y_test = references
        # 使用sklearn工具中confusion_matrix方法计算混淆矩阵
        confusion_mat = confusion_matrix(y_test, y_pred,labels=list(range(16)))
        # 使用sklearn工具包中的ConfusionMatrixDisplay可视化混淆矩阵，参考plot_confusion_matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,
            # display_labels=[
            #     "Inform","Greeting","Sighing","Questioning",
            #     "Rhetorical","Requring","Begging", "Begging",
            #     "Commissive","Desiring","Guessing","Rogering",
            #     "Agreeing","Denying","Condition","Transition"
            # ]
        )
        disp.plot(
            include_values=True,            # 混淆矩阵每个单元格上显示具体数值
            cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
            ax=None,                        # 同上
            xticks_rotation="horizontal",   # 同上
            values_format="d"               # 显示的数值格式
        )
        # plt.savefig(args.output_dir+'ConfusionMatrixDisplay.png')
        return {
            "micro_f1": micro_f1,
        }