<h2 align="center">✨AEMVC: Mitigate Imbalanced Embedding Space in Multi-view Clustering</h2>


<p align="center">
  <b>Pengyuan Li<sup>1</sup>, Man Liu<sup>2</sup>, Dongxia Chang<sup>1</sup>, Yiming Wang<sup>3</sup>, Zisen Kong<sup>1</sup>, Yao Zhao<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Institute of Information Science, Beijing Jiaotong University, Beijing, China<br>
  <sup>2</sup>School of Artificial Intelligence, Anhui University, Hefei, China<br>
  <sup>3</sup>School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, China<br>
</p>

<p align="center">
  <!-- ACM MM Badge -->
  <a href="https://dl.acm.org/doi/10.1145/3746027.3754697" target="_blank">
    <img src="https://img.shields.io/badge/ACM%20MM-2025-blueviolet.svg?style=flat-square" alt="ACM MM Proceeding">
  </a>
  <!-- arXiv Badge -->
  <!-- <a href="https://arxiv.org/abs/2412.08345" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2412.08345-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a> -->
  <!-- Contact Badge -->
  <a href="pengyuanli@bjtu.edu.cn" target="_blank">
    <img src="https://img.shields.io/badge/Email-pengyuanli%40bjtu.edu.cn-blue.svg" alt="Contact Author">
  </a>
</p>

<p align="center">
  🔥 Our work has been accepted by ACM MM 2025!<br>
</p>

## Overview🔍
<div>
    <img src="https://lummer-li.github.io/images/AEMVC.png" width="96%" height="96%">
</div>

**Figure 1. The framework of the proposed AEMVC.**


**_Abstract -_** Multi-view clustering (MVC) has gained extensive attention for its capacity to handle heterogeneous data. However, current autoencoder-based MVC methods suffer from a limitation: embedding space exhibits severe imbalances in the efficacy of feature direction, creating a long-tailed singular value distribution where few directions dominate. To mitigate this, we introduce a novel Activate-Then-Eliminate Strategy for Multi-View Clustering (AEMVC), inspired by the observation that balanced feature directions can facilitate enhancing discrimination of learned representations. AEMVC dynamically adjusts the contributions of different feature directions through two keys: a Feature Activation Module that narrows singular value discrepancies to prevent dominant directions from controlling clustering decisions, and an Inter-view Mutual Supervision strategy that filters redundant information by adaptively determining view-specific thresholds based on cross-view consistency. By activating more feature directions and eliminating each view's adverse factors, AEMVC achieves more balanced and discriminative embedding representations. Extensive experiments on seven multi-view benchmarks validate AEMVC's effectiveness, demonstrating substantial improvements over state-of-the-art methods.

## Datasets📚
To verify the performance and general applicability of our ConDSeg in the field of medical image segmentation, we conducted experiments on five challenging public datasets: Kvasir-SEG, Kvasir-Sessile, GlaS, ISIC-2016, and ISIC-2017, covering subdivision tasks across three modalities. 

Seven benchmark multi-view datasets are utilized to evaluate the performance of our AEMVC, including MSRCV1, Synthetic3d, UCI-Digit, ALOI, Handwritten, Scene15, and Animal.


| Dataset      | Samples  | Views  | View Dimensions         | Clusters  |
|--------------|----------|--------|-------------------------|-----------|
| MSRCV1       | 210      | 6      | 1302/48/512/100/256/210 | 7         |
| Synthetic3d  | 600      | 3      | 3/3/3                   | 3         |
| UCI-Digit    | 2000     | 3      | 216/76/64               | 10        |
| ALOI         | 1079     | 4      | 64/64/77/13 | 1279      | 10        |
| Handwritten  | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| Scene15      | 4485     | 3      | 20/59/40 | 2750         | 15        |
| Animal       | 11673    | 4      | 2689/2000/2001/2000     | 20        |



## Experimental Results🏆


**Table 1. Clustering results on seven multi-view datasets. “Incomplete” represents data with 50% missing views, while “Complete” denotes data without missing views. The best and second-best results are highlighted in bold and underlined, respectively. Note that VITAL, SCMVC, and DCMVC are evaluated only on complete data as they are not designed for incomplete scenarios.**
<div>
    <img src="assets/tab1.jpg" width="80%" height="96%">
</div>

<br> </br>

<div>
    <img src="assets/fig1.jpg" width="96%" height="96%">
</div>

**Figure 1. Clustering performance under different missing rates on the Handwritten dataset. The colored regions denote the standard variances with five random experiments.**


## Getting Started🚀
### Data Preparation
The dataset should be organised as follows, taking MSRCV1 as an example:
```text
MSRCV1
├── X
│   ├── X1
│   ├── X2
│   ├── X3
│   ├── ...
├── Y
```

### Training and Evaluation
- To train the AEMVC, run: `main.py`. The prediction results obtained using the K-Means algorithm.



## Cite our work📝
```bibtex
@inproceedings{li2025aemvc,
  author = {Li, Pengyuan and Liu, Man and Chang, Dongxia and Wang, Yiming and Kong, Zisen and Zhao, Yao},
  title = {AEMVC: Mitigate Imbalanced Embedding Space in Multi-view Clustering},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3754697},
  doi = {10.1145/3746027.3754697},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {6461–6470},
  numpages = {10},
  keywords = {deep multi-view clustering, feature activation, multi-view representation learning, redundancy},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```

## License📜
The source code is free for research and educational use only. Any commercial use should get formal permission first.



