
# SHGN for ROCStories Corpus

## Requirements
- pytorch-lightning == 0.8.5
- transformers >= 4.15.0
- torch >= 1.6
- torch_geometric

## Data
ROCStories Corpus can be found [here](https://github.com/JianGuanTHU/StoryEndGen/tree/master/data)

## Train and Test
For training, you can run commands like this:
```
python main.py
```
For evaluation, the command may like this:
```
python main.py --test
```


## Acknowledgement
Thanks for the following wonderful work that inspired us:
- [DHGN](https://github.com/xcfcode/DHGN) (CCL'21)
- [SimCSE](https://github.com/princeton-nlp/SimCSE) (EMNLP'21)
