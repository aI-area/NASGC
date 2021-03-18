## Overview
Here we provide the implementation of the NAS-GC in TensorFlow, along with a
 execution example (on the Cora dataset). The repository is organised as follows:
+ `data/` contains the graph dataset file (Cora);
+ `evaluator/` contains the evaluations of clustering (`metrics.py`);
+ `model/` contains:
  + evaluating the performance of NAS-GC network (`cluster.py`);
  + the implementation of self-supervision in loss function (`loss_estimator.py`);
  + the implementation of the NAS-GC network (`repres_learner.py`);
  + preparing to train NAS-GC network (`trainer.py`);
+ `util/` contains data preprocessing of graph dataset (`data_processor.py`).

Finally, `main.py` sets all hyperparameters and may be used to execute a full training run on graph dataset.

```bash
$ python main.py
```

## Architecture

![image](https://github.com/aI-area/NASGC/blob/main/framework.png)

## Dataset
| Name | Nodes | Edges | Features | Classes |
| :--: | :---: | :---: | :------: | :-----: |
| Cora | 2,708 | 5,429 |  1,433   |    7    |
| Citeseer | 3,327 | 4,732 |  3,703   |    6    |
| Pubmed | 19,717 | 44,338 |  500   |    3    |
| Wiki | 2,405 | 17,981 |  4,973   |    17    |
| ogbn-arxiv | 169,343 | 1,166,243 |  128   |    40    |

Tip: Due that the data size is too large, we cannot upload it. But it can be downloaded from https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv.

## Dependencies
The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):
+ `numpy==1.18.1`
+ `scipy==1.2.0`
+ `tensorflow-gpu==1.14.0`
+ `scikit-learn==0.22.1 `

In addition, CUDA 10.0 and cuDNN 7.5 have been used.

## References

You may also be interested in the related articles：

+ Attributed Graph Clustering via Adaptive Graph Convolution [AGC]( https://github.com/karenlatong/AGC-master)
+ Structural Deep Clustering Network [SDCN](https://github.com/bdy9527/SDCN)
+ Text-associated DeepWalk [TADW](https://github.com/benedekrozemberczki/TADW)

## License

MIT
