# Open Domain Question Answering with Explain Anything Like I'm Five

This repo contains my re-implementation of the 
["Explain Anything Like I'm Five" blog post]( https://yjernite.github.io/lfqa.html) written by 
[Yacine Jernite](https://yjernite.github.io/). I did not come up with this approach, or fit the
models myself, however I wanted to work through the steps of setting one of these QA systems up
and clearly understanding each step. This is the result. Hopefully the code and notebooks are 
useful to someone who is just starting to learn about these systems, as I was. 

I chose this blog post after I became very interested in the subject of question answering,
particularly involving the generation of a response, rather than extracting a passage from text.
There are more state-of-the art approaches out there (such as RAG) out there, but they tend to involve
very large models and knowledge bases (RAG takes 70GB). This blog post appealed to me intellectually because of
the clear and direct approach taken and practically because the models and knowledge base are smaller
and could easily fit on my linux box. 

## How it works

At a high level, this system takes the text of a wikipedia dump and cuts it up into shorter passages.
These passages are then embedded into a 128 dimensional space using a small BERT model called RETRIEBERT. 
This model has two embedding heads, one for embedding passages and one for embedding questions. It was fit
so that questions and relevant passages are close in the embedding space. These embedded passages are then
used to make a faiss index. Faiss is a library from facebook that performs
super fast vector similarity search, as in billions of vectors searched.

When a question is asked, it is embedded and then faiss is used to find the passages most similar in the
embedding space. These passages are then used as context for a BART-large type sequence to sequence model.
The question and contextual passages are concatenated and then used as input into the BART model 
which then generates a long form response to the question. 

The responses are sometimes on point, sometimes off the wall ... but they are almost always topical. It's a fun 
system to play around with.

## Installation

I did all this on my linux box. You will absolutely need a GPU to perform the wikipedia embedding which took me about 
14 hours on an Nvidia 2070 Super.

Make a virtual environment and then install the following packages like so:
~~~
pip install jupyter numpy faiss_gpu torch transformers datasets.
~~~~

Then clone or fork the repo and put it wherever you like. That's basically it.

## How to use this repo

If you just want to run this as quick as possible and play with it, you can use the 
[notebooks\qa_quick_start.ipynb](notebooks/qa_quick_start.ipynb) jupyter notebook. 
This notebook imports code in [src\qa_utils.py](src/qa_utils.py) in order
create the embeddings and also answer questions. Embedding wikipedia with the RETRIEBERT model
took me about 14 hours using a Nvidea 2070 GPU. I don't provide the embeddings in this repo because they take over
8GB of memory.

If you want to understand what's going on under the hood you can look at the 
[notebooks\qa_step_by_step.ipynb](notebooks/qa_step_by_step.ipynb) notebook. This works through the process of 
setting up the QA system step by step. It borrows **a lot** from the original 
[blog post]( https://yjernite.github.io/lfqa.html) and code.  It does (I think) perhaps explain
some of the steps more clearly to someone who is newer at this.  I learned a lot working through this
at least.

Feel free to reach out with comments, questions and/or corrections.

