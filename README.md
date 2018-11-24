# A Span Selection Model for Semantic Role Labeling (Under Construction)

## Citation
* A Span Selection Model for Semantic Role Labeling
* Hiroki Ouchi (RIKEN AIP/Tohoku Univ.), Hiroyuki Shindo (NAIST) and Yuji Matsumoto (NAIST)
* In EMNLP 2018
* Conference paper: http://aclweb.org/anthology/D18-1191
* arXiv version: https://arxiv.org/abs/1810.02245
```
@InProceedings{D18-1191,
  author = 	"Ouchi, Hiroki
		and Shindo, Hiroyuki
		and Matsumoto, Yuji",
  title = 	"A Span Selection Model for Semantic Role Labeling",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1630--1642",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1191"
}
```


## Prerequisites
* [python3](https://www.python.org/downloads/)
* [Theano](http://deeplearning.net/software/theano/)
* [h5py](https://www.h5py.org/)

## Installation
```
conda install -c conda-forge theano
conda install -c anaconda h5py
conda install mkl-service
```

## Training on GPU
Attention: Only support cuda-8.0
### Theano configuration (.theanorc)
```
# cuda inatall dir
[cuda]
root=/usr/local/cuda-8.0

[dnn]
enable=True
include_path=/usr/local/cuda-8.0/include
library_path=/usr/local/cuda-8.0/lib64

[lib]
cnmem = 0.8

[global]
device = cuda
floatX = float32
mode=FAST_RUN

[nvcc]
fastmath = True
```

## Data
### CoNLL-2005
* [Treebank-2](https://catalog.ldc.upenn.edu/LDC95T7)
### CoNLL-2012
* [OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
* We create the dataset by following the process described at http://cemantix.org/data/ontonotes.html
### Word Representations
* [SENNA](https://ronan.collobert.com/senna/download.html)
1. Download SENNA from the website
2. Run bash command in path /senna to create the final embedding file
```
paste hash/words.lst embeddings/embeddings.txt > emb.txt
```
* [ELMo](https://github.com/allenai/allennlp/tree/v0.6.1)

### Data Format
#### SENNA embedding
```
// SENNA embedding (50 dimension, 130,000 words in total)
love    0.950697 0.235026 0.359608 ... 0.61339
love-hate       1.91746 1.63426 0.530458 ... 0.459215
```
#### CoNLL-2005 Training & Development Sets
```
// 0:WORD 1:POS 2:PARSE 3:NE 4:FRAME 5:LEMMA 6-:ARGS
Ms. NNP      (S1(S(NP*         *    -   -       (A0*
Haag NNP             *)    (LOC*)   -   -          *)
plays VBZ         (VP*         *    02  play     (V*)
Elianti NNP       (NP*))       *    -   -       (A1*)
. .                  *))       *    -   -          *
```
#### CoNLL-2005 Test Set (Not including FRAME ID)
```
0:WORD 1:POS 2:PARSE 3:NE 4:LEMMA 5-:ARGS
The                DT     (S1(S(NP*         *                 (A1*
finger-pointing    JJ             *)        *    -               *)
has                AUX         (VP*         *    -               *
already            RB        (ADVP*)        *    -        (AM-TMP*)
begun              VBN         (VP*))       *    begin         (V*)
.
```
#### CoNLL-2012
```
// 0:DOCUMENT 1:PART 2:INDEX 3:WORD 4:POS 5:PARSE 6:LEMMA 7:FRAME 8:SENSE 9:SPEAKER 10:NE 11-N:ARGS N:COREF
bc/cctv/00/cctv_0001   0   0           This    DT  (TOP(S(NP*         -    -   -   Speaker#1        *   (ARG2*   (61
bc/cctv/00/cctv_0001   0   1            map    NN           *)        -    -   -   Speaker#1        *        *)   61)
bc/cctv/00/cctv_0001   0   2      reflected   VBD        (VP*    reflect  01   1   Speaker#1        *      (V*)    -
bc/cctv/00/cctv_0001   0   3            the    DT        (NP*         -    -   -   Speaker#1        *   (ARG1*     -
bc/cctv/00/cctv_0001   0   4       European    JJ           *         -    -   -   Speaker#1    (NORP)       *     -
bc/cctv/00/cctv_0001   0   5    battlefield    NN           *         -    -   -   Speaker#1        *        *     -
bc/cctv/00/cctv_0001   0   6      situation    NN           *))       -    -   -   Speaker#1        *        *)    -
bc/cctv/00/cctv_0001   0   7              .     .           *))       -    -   -   Speaker#1        *        *     -
```
[Attention] 
1. Samples should be separated by an empty line
2. Remember to remove illegal lines in CoNLL-2012 dataset, e.g.
#Start of Document, #end of document


## Usage
### Training
SENNA: `python src/main.py --mode train --train_data path/to/conll2005.train.txt --dev_data path/to/conll2005.dev.txt --data_type conll05 --drop_rate 0.1 --reg 0.0001 --hidden_dim 300 --n_layers 4 --halve_lr --word_emb path/to/senna --save --output_dir output`

ELMo: `python src/main.py --mode train --train_data path/to/conll2005.train.txt --dev_data path/to/conll2005.dev.txt --data_type conll05 --drop_rate 0.1 --reg 0.0001 --hidden_dim 300 --n_layers 4 --halve_lr --train_elmo_emb path/to/elmo.conll2005.train.hdf5 --dev_elmo_emb path/to/elmo.conll2005.dev.hdf5 --save --output_dir output`

### Predicting
SENNA: `python src/main.py --mode test --test_data path/to/conll2005.test.txt --data_type conll05 --drop_rate 0.1 --hidden_dim 300 --n_layers 4 --output_dir output --output_fn conll2005.test --word_emb path/to/senna --load_label output/label_ids.txt --load_param output/param.epoch-0.pkl.gz --search greedy`

ELMo: `python src/main.py --mode test --test_data path/to/conll2005.test.txt --data_type conll05 --drop_rate 0.1 --hidden_dim 300 --n_layers 4 --output_dir output --output_fn conll2005.test --test_elmo_emb path/to/elmo.conll2005.test.hdf5 --load_label output/label_ids.txt --load_param output/param.epoch-0.pkl.gz --search greedy`


## LICENSE
MIT License
