# AwesomeSEG
This repo contains our research summary for Story Ending Generation (SEG), and we also provide the codes and generated results of our work on SEG.   

Contents:  
1. [SEG Paper List](https://github.com/krystalan/AwesomeSEG#1seg-task)
2. [SHGN (DASFAA'22)](https://github.com/krystalan/AwesomeSEG#2shgn)

## 1.SEG Task
Story ending generation is the task of generating an ending sentence of a story given a story context. For example, given the story context:
```
Today is Halloween. 
Jack is so excited to go trick or treating tonight.
He is going to dress up like a monster.
The costume is real scary.
```
We hope the SEG model could generate a reasonable ending for the above story, such as:  
```
He hopes to get a lot of candy.
```

### 1.1 Dataset - ROCStories Corpus
Existing SEG works all utilize **ROCStories Corpus** to evaluate performances of SEG model. Specifically, the **ROCStories Corpus** contains 98,162 five-sentence stories, in which the first four sentences is used as story context while the last one is regarded as story ending sentence.

### 1.2 Existing Work
| Paper | Conference/Journal | Results (BLEU-1/2) | Evaluation Tools | Code | Tags |
| :--: | :--: | :--: | :--: | :--: | :--: |
| [From Plots to Endings: A Reinforced Pointer Generator for Story Ending Generation](https://arxiv.org/abs/1901.03459) | NLPCC 2018 | 28.51/11.92 | [nlg-eval](https://github.com/Maluuba/nlg-eval) | [SEG](https://github.com/blcunlp/SEG) ![](https://img.shields.io/github/stars/blcunlp/SEG.svg?style=social) |`arch-LSTM`、`train-MLE`|
| [Generating Reasonable and Diversified Story Ending Using Sequence to Sequence Model with Adversarial Training](https://aclanthology.org/C18-1088/) | COLING 2018 | - | - | `arch-LSTM`、`train-GAN`、`train-MLE` |
| [WriterForcing: Generating more interesting story endings](https://aclanthology.org/W19-3413/) | ACL 2019 @ Storytelling | - | - | [WriterForcing](https://github.com/witerforcing/WriterForcing) ![](https://img.shields.io/github/stars/witerforcing/WriterForcing.svg?style=social) | `arch-GRU`、`info-Keywords`、`train-MLE`、`train-ITF` |
| [Learning to Control the Fine-grained Sentiment for Story Ending Generation](https://aclanthology.org/P19-1603/) | ACL 2019 Short | 19.8/6.7 | - | [sentimental-story-ending](https://github.com/Hunter-DDM/sentimental-story-ending) ![](https://img.shields.io/github/stars/Hunter-DDM/sentimental-story-ending.svg?style=social) | `arch-LSTM`、`info-Sentiment`、`train-MLE` |
| [Story Ending Generation with Incremental Encoding and Commonsense Knowledge](https://arxiv.org/abs/1808.10113) | AAAI 2019 | 26.82/9.36 | - | [StoryEndGen](https://github.com/JianGuanTHU/StoryEndGen) ![](https://img.shields.io/github/stars/JianGuanTHU/StoryEndGen.svg?style=social) | `arch-LSTM`、`info-knowledge`、`train-MLE` |
| [Generating Diverse Story Continuations with Controllable Semantics](https://aclanthology.org/D19-5605/) | EMNLP 2019 @ NGT | - | - | - | `arch-LSTM`、`info-Controllable`、`train-MLE` |
| [Toward a Better Story End: Collecting Human Evaluation with Reasons](https://aclanthology.org/W19-8646/) | INLG 2019 | - | - | [SEG_HumanEvaluationReasons](https://github.com/mil-tokyo/SEG_HumanEvaluationReasons) ![](https://img.shields.io/github/stars/mil-tokyo/SEG_HumanEvaluationReasons.svg?style=social) | `task-Metric` |
| [Story Ending Generation with Multi-Level Graph Convolutional Networks over Dependency Trees](https://ojs.aaai.org/index.php/AAAI/article/view/17545) | AAAI 2021 | 24.6/8.6 | - | [MLGCN-DP](https://github.com/VISLANG-Lab/MLGCN-DP) ![](https://img.shields.io/github/stars/VISLANG-Lab/MLGCN-DP.svg?style=social) | `arch-LSTM`、`arch-GCN`、`info-DP`、`train-MLE` |
| [Incorporating sentimental trend into gated mechanism based transformer network for story ending generation](https://www.sciencedirect.com/science/article/abs/pii/S0925231221000618) | Neurocomputing 2021 | 27.03/7.62 | - | - |`arch-Transformer`、`info-Sentiment`、`train-MLE` |
| [Incorporating Commonsense Knowledge into Story Ending Generation via Heterogeneous Graph Networks](https://arxiv.org/abs/2201.12538) | DASFAA 2022 | 25.6/9.4 | nlg-eval & py-rouge | [SHGN](https://github.com/krystalan/AwesomeSEG) | `arch-Transformer`、`arch-gat`、`info-Sentiment`、`info-knowledge` |

The concepts used in Tags are illustrated as follows:  
- arch：The architecture of the model, includes `arch-LSTM`、`arch-GRU`、`arch-Transformer` and `arch-GCN` tags.
- train：The training strategy of the model, includes `train-MLE`、`train-GAN` and `train-ITF` tags.
- info：The additional infomation used in SEG, includes `info-Keywords`、`info-Sentiment`、`info-knowledge`、`info-DP` (Dependency Parsing) and `info-Controllable` tags.
- task：`task-Metric` tag indicates the evaluation work.

## 2.SHGN
We provide the codes and generated results of the DASFAA 2022 paper [*Incorporating Commonsense Knowledge into Story Ending Generation via Heterogeneous Graph Networks*](https://arxiv.org/abs/2201.12538).

### Codes
Please refer to the [SHGN](https://github.com/krystalan/AwesomeSEG/tree/main/SHGN) directory.

### Generated results
The generated results of our SHGN are available at [SHGN.txt](https://github.com/krystalan/AwesomeSEG/blob/main/results/SHGN.txt). To reproduce the evaluation scores of our paper, please use [nlg-eval](https://github.com/Maluuba/nlg-eval) and [py-rouge](https://github.com/Diego999/py-rouge) toolkits to calculate BLEU and ROUGE scores, respectively.
