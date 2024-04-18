# MiniGPT
## Introduction

This project explores the implementation of a mini-GPT model, a simplified version of the powerful GPT-2 architecture.  It demonstrates the implementation of a generative pre-trained transformer model capable of text generation at a character level.  We use character-based Shakespeare dataset to generate a Shakespeare's-like text. This project covers aspects from data preparation, tokenization, model building, training, to generating new text based on trained models.




## Table of Contents

- [Environment setup](#environment-setup)
- [Dependency installation](#dependency-installation)
- [Instructions to run the code](#instructions-to-run-the-code)
- [Results and performance metrics](#performance-metrics-and-results)
- [Observations and findings](#observations-and-findings)
- [Reflection](#reflection)


## Environment setup

To run this project, ensure you have Python 3.x installed. 

## Dependency installation

```
pip install torch numpy random matplotlib math regex
```

Dependencies:

- [pytorch](https://pytorch.org) 
- [numpy](https://numpy.org/install/)
- [matplotlib](https://matplotlib.org/)
- [math](https://docs.python.org/3/library/math.html)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [datasets](https://huggingface.co/docs/datasets/index)
- `regex`for data tokenization with regularization



## Instructions to run the code

- [`./utils_gpt.ipynb`](/utils_gpt.ipynb) - All the class and functions we constracted.
- [`./explore_data.ipynb`](/explore_data.ipynb) - Exploring the Shakespeare data set.
- [`./Compare_tokenizations.ipynb](/Compare_tokenizations.ipynb)- Comparison of three tokenization methods.
- [`./vocabulary](/vocabulary) - The vocabulary we saved.
- [`./mini_gpt_training.ipynb](/mini_gpt_training.ipynb) -Tuning hyperparameters
- [`./Best_model.ipynb](/Best_model.ipynb) - The best model
- [`./minigpt_fine_tuning.ipynb](/minigpt_fine_tuning.ipynb) - Fine-tuning
  



## Performance metrics and Results 
### Comparison within different tokenization methods
Naive Tokenization:

<img src="image/token1.png" width="600">
BPE:

<img src="image/token2.png" width="600">
BPE with Regularization:

<img src="image/token3.png" width="600">

### Hyperparameters tuning
We use cross-validation loss and human evaluation scores to evaluate our model.

After tuning hyperparameters, we obtained the best hyperparameters as follows:
| embeddings | layers | heads | batch size | block size | learning rate | drop out |
|------------|--------|-------|------------|------------|---------------|----------|
| 488        | 10     | 8     | 16         | 32         | 0.0001        | 0.1      |

In the following is the generated text with the best hyperparameters
<img src="image/generated_text.png" width="600">
<img src="image/loss_plot_best.png" width="600">

## Observations and findings
- **Vocabulary Size Limitations**: Due to hardware constraints, we set the vocabulary size to only 3,257, which is significantly smaller than GPT-2's 50,257. This limitation may reduce our model's accuracy in semantic understanding and the quality of text generation, especially for complex or linguistically rich content.
- **Number of Iterations**: We capped the number of iterations at 5,000 due to hardware limitations. This restriction suggested the model was undertrained as both training and validation losses continued to decrease at this threshold, indicating potential missed optimal parameters.
- **Learning Rate Warm-up**: We found that starting with a very low learning rate and gradually increasing it helped stabilize the training. This approach improved the model's performance compared to using a fixed rate, which led to instability or slow convergence.
- **Text Evaluation Metrics**: We relied on validation loss and subjective human evaluation for assessing text quality because implementing BLEU scores proved challenging. The dependency of BLEU scores on the quality and comprehensiveness of reference texts was a limiting factor.
- **Early Stopping**: We employed early stopping to prevent overfitting, but this technique sometimes halted the training too soon, as evidenced by poorer performance of text generated under early stopping conditions compared to slightly overfit models.
- **Comparison with Other Algorithms**: We highlighted the advantages of the Transformer architecture used in GPT-2 over traditional algorithms like RNNs and LSTMs, particularly in handling long-distance dependencies and training efficiency through parallel processing.
- **Ethics and Privacy**: We addressed potential misuse of the model for creating false information, copyright infringement risks, and socio-cultural biases in the training data. To mitigate these risks, we took measures to ensure data diversity and removed sensitive content from our datasets.

### In hyperparameters tuning

### In fine-tuning based on Transfer Learning
Pretraining on a large dataset and fine-tuning on a small dataset seems to yield better results than pretraining on a small dataset and fine-tuning on a large one. A larger amount of data during pretraining enhances the model's generalization ability, which better facilitates the transfer to fine-tuning on a smaller dataset. However, due to significant differences in the distribution between the two datasets, freezing only the last linear layer does not yield good results. Fine-tuning only one layer of parameters is insufficient.

Although the performance of our fine-tuning based on the pre-trained model is not as ideal as we hoped, considering the long training time when training models, we can use fine-tuning when we want to accomplish a new task without wasting more time training an entire model from scratch. Fine-tuning allows the model to adapt to new tasks while retaining useful knowledge already learned, which is often much more efficient than training a completely new model from scratch.

## Reflection
### Running time issue:
Small vocabulary size of BPE
limit number of iterations
### Choosing Evaluation Metrics 
