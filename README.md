# MiniGPT
## Introduction

This project explores the implementation of a mini-GPT model, a simplified version of the powerful GPT-2 architecture.  It demonstrates the implementation of a generative pre-trained transformer model capable of text generation at a character level.  We use character-based Shakespeare dataset to generate a Shakespeare's-like text. This project covers aspects from data preparation, tokenization, model building, training, to generating new text based on trained models.


<img src="image/text_generation_shakespeare_rnn.jpg" width="400">


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
- [`./Compare_tokenizations.ipynb](/Compare_tokenizations.ipynb)- Comparison of three tokenization methods 
- [`./minigpt/sidecar.py`](/minigpt/sidecar.py) - Training utilities such as
  W&B logging, auto-saving etc.
- [`./scripts/*.py`](/scripts/) - Scripts exposing CLIs to interact with MiniGPT.

## Performance metrics and Results 
### Cross-Validation Loss
The formula is as follows:

The Cross-Validation Loss is calculated as follows:

\[ \text{Cross-Validation Loss} = -\sum_{i=1}^{C} y_{i} \log(\hat{y}_{i}) \]

where \(C\) is the token vocabulary size, \(y_{i}\) is the actual probability of the \(i\)-th token, and \(\hat{y}_{i}\) is the predicted probability of the \(i\)-th token.


In the following is the generated text with the best hyperparameters
<img src="image/generated_text.png" width="600">

## Observations and findings
### In hyperpameters tuning

### In fine-tuning based on Transfer Learning
Pretraining on a large dataset and fine-tuning on a small dataset seems to yield better results than pretraining on a small dataset and fine-tuning on a large one. A larger amount of data during pretraining enhances the model's generalization ability, which better facilitates the transfer to fine-tuning on a smaller dataset. However, due to significant differences in the distribution between the two datasets, freezing only the last linear layer does not yield good results. Fine-tuning only one layer of parameters is insufficient.

Although the performance of our fine-tuning based on the pre-trained model is not as ideal as we hoped, considering the long training time when training models, we can use fine-tuning when we want to accomplish a new task without wasting more time training an entire model from scratch. Fine-tuning allows the model to adapt to new tasks while retaining useful knowledge already learned, which is often much more efficient than training a completely new model from scratch.

## Reflection
### Running time issue:
Due to the hardware limitation,the training time for our model is quite long. Consequently, we set the vocabulary size of our Tokenization of BPE with regularization to only 3,257. In the official GPT-2 model, the vocabulary size can reach 50,257. However, considering our smaller dataset and its specific focus on generating Shakespeare-style text, we opted for a vocabulary size of 3,257.

As a consequence, using a small vocabulary in our model may potentially reducing the model's accuracy in semantic understanding and generation when dealing with complex texts or linguistically rich languages. And also the quality of generated text is affected because the vocabulary available for constructing sentences is limited, which may hinder the effective expression of complex ideas or maintaining textual coherence. 

Also due to hardware constraints, we set the number of iterations to 5,000. However, during the fine-tuning process, we observed that both the training loss and validation loss continued to decrease at 5,000 iterations, indicating that the model was not fully trained. This could potentially lead to missing some optimal parameters.

Therefore, if time and equipment permit, we should consider increasing the vocabulary size and iterations to better optimize our model.

### Choosing Evaluation Metrics 
At the begining, we are quite confused which evaluation metrics should be used. And we are confused with the relationship between validation loss and the quanlity of generated text because sometimes a low validation loss doesn't always indicate good quanlity of generated text. Therefore, in the end, we used a combination of validation loss and human evaluation. However, since we are not experts in Shakespearean literature, our scoring may be quite subjective. 

There are other text evaluation metrics such as BLEU. BLEU measures how close the machine-generated translations are to one or more reference translations. The core idea is to quantify the matches between the candidate translation produced by the machine and the reference translations at the level of words and phrases. 

It is also hard for us to implement BLEU because the BLEU score is highly dependent on the selected reference texts. If the number of reference texts is limited or their quality is low, they may not comprehensively cover all effective expressions, leading to biases in the evaluation results. Moreover, for tasks that are inherently open-ended and subjective, finding a comprehensive set of references is particularly challenging. Therefore, we did not choose BLEU as our evaluation metrics.
