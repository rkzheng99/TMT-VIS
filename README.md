# TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation (NeurIPS 23)
[Rongkun Zheng](https://rkzheng99.github.io), [Lu Qi](http://luqi.info/), [Xi Chen](https://xavierchen34.github.io/), [Yi Wang](https://shepnerd.github.io/), Kun Wang, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN&oi=ao), [Hengshuang Zhao*](https://hszhao.github.io/)

[[paper]](https://arxiv.org/abs/2312.06630) [[code]](https://github.com/rkzheng99/TMT-VIS) 
![image](https://github.com/rkzheng99/TMT-VIS/blob/main/img/model.png)

## Highlights
- Our paper was accepted by NeurIPS 2023 (poster)!

## Abstract
Training on large-scale datasets can boost the performance of video instance segmentation while the annotated datasets for VIS are hard to scale up due to the high labor cost. What we possess are numerous isolated filed-specific datasets, thus, it is appealing to jointly train models across the aggregation of datasets to enhance data volume and diversity. However, due to the heterogeneity in category space, as mask precision increases with the data volume, simply utilizing multiple datasets will dilute the attention of models on different taxonomies. Thus, increasing the data scale and enriching taxonomy space while improving classification precision is important. In this work, we analyze that providing extra taxonomy information can help models concentrate on specific taxonomy, and propose our model named **T**axonomy-aware **M**ulti-dataset Joint **T**raining for **V**ideo **I**nstance **S**egmentation (**TMT-VIS**) to address this vital challenge. Specifically, we design a two-stage taxonomy aggregation module that first compiles taxonomy information from input videos and then aggregates these taxonomy priors into instance queries before the transformer decoder. We conduct extensive experimental evaluations on four popular and challenging benchmarks, including YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and UVO. Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all benchmarks. These appealing and encouraging results demonstrate the effectiveness and generality of our approach.

## Experimental Results
Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all benchmarks.
### Youtube-VIS 2019
| Method          | Backbone | AP   | AP<sub>50</sub> | AP<sub>75</sub> |
|:---------------:|----------|------|---------|---------|
| Mask2Former-VIS | Swin-L   | 60.4 | 84.4    | 67.0    |
| VITA            | Swin-L   | 63.0 | 86.9    | 67.9    |
| IDOL            | Swin-L   | 64.3 | 87.5    | 71.0    |
| **TMT-VIS**         | Swin-L   | 65.4 | 88.2    | 72.1    |
### Youtube-VIS 2021
| Method          | Backbone | AP   | AP<sub>50</sub> | AP<sub>75</sub> |
|:---------------:|----------|------|---------|---------|
| Mask2Former-VIS | Swin-L   | 52.6 | 76.4    | 57.2    |
| VITA            | Swin-L   | 57.5 | 80.6    | 61.0    |
| IDOL            | Swin-L   | 56.1 | 80.8    | 63.5    |
| **TMT-VIS**         | Swin-L   | 61.9 | 82.0    | 68.3    |
### UVO
| Method          | Backbone | AP   | AP<sub>50</sub> | AP<sub>75</sub> |
|:---------------:|----------|------|---------|---------|
| Mask2Former-VIS | Swin-L   | 27.3 | 42.0    | 27.2    |
| **TMT-VIS**         | Swin-L   |29.9  | 43.6    | 30.1   |
### OVIS
| Method          | Backbone | AP   | AP<sub>50</sub> | AP<sub>75</sub> |
|:---------------:|----------|------|---------|---------|
| Mask2Former-VIS | Swin-L   | 23.1 | 45.4    | 21.8    |
| VITA            | Swin-L   | 27.7 | 51.9    | 24.9    |
| IDOL            | Swin-L   | 42.6 | 65.7    | 45.2    |
| **TMT-VIS**         | Swin-L   | 46.9 | 71.0    | 48.9    |

## Citation
If you find this work is useful for your research, please cite our papers:
```
@inproceedings{zheng2023tmtvis,
  title={{TMT}-{VIS}: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation},
  author={Rongkun, Zheng and Lu, Qi and Xi, Chen and Yi, Wang and Kun, Wang and Yu, Qiao, and Hengshuang, Zhao},
  booktitle={NeurIPS},
  year={2023}
}
```
## Acknowledgement
This work is partially supported by the National Natural Science Foundation of China (No. 62201484), National Key R&D Program of China (No. 2022ZD0160100), HKU Startup Fund, and HKU Seed Fund for Basic Research.

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), and [VITA](https://github.com/sukjunhwang/VITA).Thanks for their excellent works.

