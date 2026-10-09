# Learning-CDF: Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation

This repository contains the implementation for the paper:  
Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation.

## Overview

Planar quadrilateral (PQ) mesh generation relies on designing suitable conjugate direction fields (CDFs) over a surface. This project implements a learning-based approach for predicting CDFs from a triangle mesh and user-provided strokes.


## Repository structure

```text
Learning-CDF/
├── README.md
├── LICENSE                 
├── dataset.py               
├── model.py                 
├── loss.py                  
├── train.py
├── test.py                 
├── Configs/
    ├── config_train.yaml
    └── config_test.yaml   
└── utils/
    ├── __init__.py
    ├── config.py             
    ├── data_loader.py        
    └── data_utils.py       
```

## Dataset

This project requires a preprocessed dataset containing triangle meshes with geometric feature, ground-truth conjugate direction fields (CDFs), and streamlines traced from the CDFs.

### Dataset structure

The dataset will be organized as follows:

```text
Datasets/
├── TriMesh_train.pkl
├── Streams_train.pkl
├── TriMesh_valid.pkl
├── Streams_valid.pkl
├── TriMesh_test.pkl
└── Streams_test.pkl
```
The dataset will be released soon. We will update this section with the download link once it becomes available.

### Dataset format

Each `TriMesh_<split>.pkl` must be a dictionary whose values are lists containing one entry per surface:

| Key | Meaning | Shape of each entry |
| --- | --- | --- |
| `names` | Surface names | string |
| `tri_vertices` | Vertex coordinates | `(V, 3)` |
| `tri_faces` | Triangle vertex indices | `(F, 3)` |
| `tri_vnormals` | Vertex normals | `(V, 3)` |
| `tri_fnormals` | Face normals | `(F, 3)` |
| `tri_cd1` |  First conjugate direction | `(F, 3)` |
| `tri_cd2` |  Second conjugate direction | `(F, 3)` |

Here, V and F denote the numbers of vertices and faces in the triangle mesh, respectively.

Each `Streams_<split>.pkl` must contain:

| Key | Meaning |
| --- | --- |
| `names` | Surface name |
| `streams` | A list per surface of streams, each represented by 3D points `(L, 3)` |
| `streams_fid` | Indices of the triangle mesh faces intersected by each stream (`L - 1` per stream) |

Here, L denotes the number of points sampled along a streamline.



## Training

Edit `Configs/config_train.yaml` to set the training dataset path, output directory, and  training hyperparameters.

Then run from the repository root:

```bash
python train.py --config ./Configs/config_train.yaml
```

## Testing

Edit `Configs/config_test.yaml` to set the testing dataset path, output directory and pretrained model checkpoint.

Then run from the repository root:

```bash
python test.py --config ./Configs/config_test.yaml
```


## Citation

If you find our work useful in your research, please cite:

```bibtex
@inproceedings{tao2026learning,
  title={Learning conjugate direction fields for planar quadrilateral mesh generation},
  author={Tao, Jiong and Yang, Yong-Liang and Deng, Bailin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={30},
  pages={25867--25876},
  year={2026}
}
```


## License
This code is released under **BSD 2-Clause License**.

## Contact 

If you have any questions about this code, please feel free to contact Jiong Tao ([jt2337@bath.ac.uk](mailto:jt2337@bath.ac.uk)), Yong-Liang Yang ([y.yang@cs.bath.ac.uk](mailto:y.yang@cs.bath.ac.uk)) or Bailin Deng ([DengB3@cardiff.ac.uk](mailto:DengB3@cardiff.ac.uk)).
