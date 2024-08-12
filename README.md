<h2 align="center">
Adap-$\tau_{vc}$: Integrating Contrastive Learning and MAD+VAR Enhancing on Adaptively Modulating Embedding Magnitude for Recommendation
</h2>

This is the PyTorch implementation. 

## Dependencies
- pytorch==1.11.0
- numpy==1.21.5
- scipy==1.7.3
- torch-scatter==2.0.9

## Training model:
- mkdir log
- mkdir output
- cd bash

#### yelp2018
```
bash Adap_tau_novel.sh  yelp2018 1e-3 1e-1 3 1024 2048 drop 1.0 1.5 no_sample 0 100 nocosine lgn weight_mean 1.2 0.1 0.25
```
#### amazon-book
```
bash Adap_tau_novel.sh amazon-book 1e-4 1e-1 3 1024 2048 nopdrop 1.0 1.0 no_sample 0 100 nocosine lgn weight_mean 0.1 1.1 0.2
```

#### gowalla
```
bash Adap_tau_novel.sh gowalla 1e-3 1e-5 3 1024 2048 nopdrop 0.8 0.6 no_sample 0 100 nocosine lgn weight_mean 0.1 1.0 0.25
```

