>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [An Accurate and Efficient Code Clone Detection
Approach for Multiple Languages](). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install javalang
pip install Levenshtein
pip install scikit-learn nltk
pip install pygments
pip install pandas
pip install scikit-learn
pip install xgboost
pip install numpy
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Feacture Extraction

To run the approach in the paper, run this command firstly:

```train
python feacture.py
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Train and Eval

To train and evaluate on dataset, run:

```eval
python classify.py 
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

[//]: # (## Pre-trained Models)

[//]: # ()
[//]: # (You can download pretrained models here:)

[//]: # ()
[//]: # (- [My awesome model]&#40;https://drive.google.com/mymodel.pth&#41; trained on ImageNet using parameters x,y,z. )

[//]: # ()
[//]: # (>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained &#40;if applicable&#41;.  Alternatively you can have an additional column in your results table with a link to the models.)

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 