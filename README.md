# Open Domain Question Answering with Explain Anything Like I'm Five

This repo contains my re-implementation of the 
["Explain Anything Like I'm Five" blog post]( https://yjernite.github.io/lfqa.html) written by 
[Yacine Jernite](https://yjernite.github.io/). 
 The original code can be found
[here](https://github.com/huggingface/notebooks/blob/master/longform-qa/lfqa_utils.py) and the
original license is [Apache 2.0](https://github.com/huggingface/notebooks/blob/master/LICENSE).

I started this project because I was learning about modern question answering systems, 
particularly those involving generating an answer rather than just extracting one from a 
provided text. I wanted to work through all the details of setting such a system up, so that
I clearly understood all the steps. This repo is the result. 

Although more state-of-the art approaches (such as RAG) exist, they tend to involve
very large models and knowledge bases (RAG takes 70GB). The EALI5 QA system uses smaller models
and a knowledge base that could easily fit on my Linux box. This blog post also appealed to me intellectually
because it was well written with a very intuitive approach. It still took me a bit to figure
all the steps out, but this was time well spent. Hopefully the code, notebooks and comments I provided here will be
useful to someone who is just starting to learn about modern QA systems, as I was at the time. 

## How it works

At a high level, this system takes the text of a wikipedia dump and cuts it up into shorter passages.
These passages are then embedded into a 128 dimensional space using a small BERT model called 
[RetriBERT](https://huggingface.co/transformers/model_doc/retribert.html). 
This model has two embedding heads, one for embedding passages and one for embedding questions. It was fit
so that questions and relevant passages are close in the embedding space. These embedded passages are then
used to make a [faiss](https://github.com/facebookresearch/faiss) index. 
Faiss is a library from facebook that performs
super fast vector similarity search over sets of billions or more vectors. 

When a question is asked, it is embedded and then faiss is used to find the passages most similar in the
embedding space. These passages are then used as context for a BART-large type sequence to sequence model.
The question and contextual passages are concatenated and then used as input into the BART model 
which then generates a long form response to the question. 

The responses are sometimes on point, sometimes off the wall ... but they are almost always topical. It's a fun 
system to play around with.

## Installation

I did all this on my linux box. You will absolutely need a GPU to perform the wikipedia embedding which took me about 
14 hours on an Nvidia 2070 Super.

Make a python 3 virtual environment and then install the following packages like so:
~~~
pip install jupyter numpy faiss_gpu torch transformers datasets
~~~~

Then clone or fork the repo and put it wherever you like. That's basically it.

## How to use this repo

If you just want to run this as quick as possible and play with it, you can use the 
[notebooks/qa_quick_start.ipynb](notebooks/qa_quick_start.ipynb) jupyter notebook. 
This notebook imports code in [src/qa_utils.py](src/qa_utils.py) in order
create the embeddings and also answer questions. Embedding wikipedia with the RETRIEBERT model
took me about 14 hours using a Nvidea 2070 GPU. I don't provide the embeddings in this repo because they take over
8GB of memory.

If you want to understand what's going on under the hood you can look at the 
[notebooks/qa_step_by_step.ipynb](notebooks/qa_step_by_step.ipynb) notebook. This works through the process of 
setting up the QA system step by step. It borrows **a lot** from the original 
[blog post]( https://yjernite.github.io/lfqa.html) and code.  It does (I think) explain
the steps more granularlly, which may be useful to those newer at this. I certainly learned
a lot working through this. 

Feel free to reach out with comments, questions and/or corrections.

