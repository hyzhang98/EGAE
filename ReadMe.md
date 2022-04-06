# Embedding Graph Auto-Encoder for Graph Clustering


This repository is our implementation of 

[Hongyuan Zhang, Pei Li, Rui Zhang, and Xuelong Li, "Embedding Graph Auto-Encoder for Graph Clustering," *IEEE Transactions on Neural Networks and Learning Systems*, 2022](https://ieeexplore.ieee.org/document/9741755).



The core idea of EGAE is to **design a GNN to find an ideal space for the relaxed *k*-means** on graph data. We prove that the relaxed *k*-means will obtain a precise clustering result under some strong assumptions. So we attempt to use GNNs to map the data into an ideal space that satisfies the strong assumptions.  



If you have issues, please email:

hyzhang98@gmail.com or hyzhang98@mail.nwpu.edu.cn.

## How to Run EGAE
```
python run.py
```
### Requirements 
pytorch >= 1.3.1

scipy 1.3.1

scikit-learn 0.21.3

numpy 1.16.5

### Remark

- model.py: An efficient implementation which can be used when datasets are not too large. 
- sparse_model.py: It is a sparse implementation of EGAE for large scale datasets, *e.g.,* PubMed. 

## Citation

```
@article{EGAE,
  author={Zhang, Hongyuan and Li, Pei and Zhang, Rui and Li, Xuelong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Embedding Graph Auto-Encoder for Graph Clustering}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TNNLS.2022.3158654}
}
```

