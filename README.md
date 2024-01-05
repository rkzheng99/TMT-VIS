# TMT-VIS: Taxonomy-aware Multi-dataset Joint Training for Video Instance Segmentation (NeurIPS 23)
[Rongkun Zheng](https://rkzheng99.github.io), [Lu Qi](http://luqi.info/), [Xi Chen](https://xavierchen34.github.io/), [Yi Wang](https://shepnerd.github.io/), Kun Wang, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN&oi=ao), [Hengshuang Zhao*](https://hszhao.github.io/)

[[paper]](https://arxiv.org/abs/2312.06630) [[code]]([https://arxiv.org/abs/2312.06630](https://github.com/rkzheng99/TMT-VIS)) 
![image](https://github.com/rkzheng99/TMT-VIS/blob/main/img/model.png)

## Highlights
- Our paper was accepted by NeurIPS 2023 (poster)!

## Abstract
Training on large-scale datasets can boost the performance of video instance segmentation while the annotated datasets for VIS are hard to scale up due to the high labor cost. What we possess are numerous isolated filed-specific datasets, thus, it is appealing to jointly train models across the aggregation of datasets to enhance data volume and diversity. However, due to the heterogeneity in category space, as mask precision increases with the data volume, simply utilizing multiple datasets will dilute the attention of models on different taxonomies. Thus, increasing the data scale and enriching taxonomy space while improving classification precision is important. In this work, we analyze that providing extra taxonomy information can help models concentrate on specific taxonomy, and propose our model named **T**axonomy-aware **M**ulti-dataset Joint **T**raining for **V**ideo **I**nstance **S**egmentation (**TMT-VIS**) to address this vital challenge. Specifically, we design a two-stage taxonomy aggregation module that first compiles taxonomy information from input videos and then aggregates these taxonomy priors into instance queries before the transformer decoder. We conduct extensive experimental evaluations on four popular and challenging benchmarks, including YouTube-VIS 2019, YouTube-VIS 2021, OVIS, and UVO. Our model shows significant improvement over the baseline solutions, and sets new state-of-the-art records on all benchmarks. These appealing and encouraging results demonstrate the effectiveness and generality of our approach.

## Experimental Results
![image](https://github.com/rkzheng99/TMT-VIS/blob/main/img/results.png)

