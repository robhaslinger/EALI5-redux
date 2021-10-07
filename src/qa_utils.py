# This file contains utility functions for implementing the QA system in the ELI5 Blog Post
# This is largely a refactoring of a subset of the code in this repository:
# https://github.com/huggingface/notebooks/blob/master/longform-qa/lfqa_utils.py
# This refactored version was created by Rob Haslinger 2021

# ----------------------------------------------------------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------------------------------------------------------

import time
import numpy as np
import faiss
import torch


# ----------------------------------------------------------------------------------------------------------------------
# Code for creating the wiki40b-snippet embeddings
# ----------------------------------------------------------------------------------------------------------------------

def embed_passage_batch(passages, tokenizer, embedding_model, max_length, device="cuda:0"):
    """
    This is a refactoring of the embed_passages_for_retrieval function in the blog code.
    :param passages: (N element list) the batch of passages to be embedded. Each element in the list is a string
    :param tokenizer: the retribert tokenizer from https://huggingface.co/yjernite/retribert-base-uncased
    :param embedding_model: the retribert model from https://huggingface.co/yjernite/retribert-base-uncased
    :param max_length: (int) the maximum number of tokens for each embedded passage. The code either truncates or pads to this length
    :param device: the device (cpu or cuda:x) that the embeddings will be calculated on

    :return: N*max_length dimension numpy array containing the batch embeddings
    """

    # the tokenization base class has (I think) been updated since the blog post.
    # The blog code throws warnings. This should work.
    tokenized_passages = tokenizer(passages,
                                   max_length=max_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors='pt')

    # now make the embeddings
    # note we are moving this to the gpu
    # we don't want to calculate gradients here because we are only doing inference
    with torch.no_grad():
        embedded_passages = embedding_model.embed_answers(tokenized_passages["input_ids"].to(device),
                                                          tokenized_passages["attention_mask"].to(device))

    # now change these to numpy and return (because we will save them as a np.memmap)
    # note if we hadn't used the torch.no_grad() we would have had to detach the gradients.
    return embedded_passages.cpu().numpy()


def embed_passages(passages_dataset, tokenizer, embedding_model, batch_size=256, n_batches=None,
                   max_length=128, index_file_name="embed.dat", dtype="float32", device="cuda:0"):
    """
    Loops over the entire wikisnippets dataset embedding each passage and then saves these to disk using a
    numpy memmap as the entire embedding is on the order of 8GB.
    :param passages_dataset: the wikisnippets (or similar) dataset
    :param tokenizer: the retribert tokenizer from  https://huggingface.co/yjernite/retribert-base-uncased
    :param embedding_model: the retribert model from https://huggingface.co/yjernite/retribert-base-uncased
    :param batch_size: embedding batch size. default is 256 based on the blog using 512 and a 16GB GPU and the current
                       work being performed on an 8GB GPU
    :param n_batches: (int) allows for testing by only embedding a small number of batches. If None, then the total
                      number of batches to span the entire dataset is calculated and used.
    :param max_length: Length of token sequence for each passage. Defaults to 128.
    :param index_file_name: (str) the filename where the embeddings are stored
    :param dtype: (str) datatype of the stored embeddings.
    :param device: (str, either 'cpu' or 'cuda:x') the device upon which the embeddings will be computed

    :return: None. The embeddings are saved direct to disk.
    """

    # put the embedding_model on the chosen device
    embedding_model.to(device)

    # find the number of batches, default behavior is to use all data.
    if n_batches is None:
        n_rows = passages_dataset.num_rows
        n_batches = int(np.ceil(n_rows / batch_size))
    else:
        n_rows = int(n_batches * batch_size)
    print("embedding", str(n_rows), "passages in", n_batches, "separate batches")

    # make the memmap to store the data in
    fp = np.memmap(index_file_name, dtype=dtype, mode='w+', shape=(n_rows, max_length))

    # now loop over all the passages
    start_time = time.time()
    for i in range(n_batches):

        # get the next batch of passages
        # note apparently indexing a data set beyond the number of rows in a dataset will just return
        # the whatever rows we have and not throw an error ... shrug.
        passage_batch = [p for p in passages_dataset[i * batch_size: (i + 1) * batch_size]["passage_text"]]

        # update message
        if i % 100 == 0:
            print("embedding batch", str(i), "of", str(n_batches), "batches", len(passage_batch), "passages in batch",
                  "at time", str(time.time() - start_time))

        # embed the passages
        embedded_batch = embed_passage_batch(passage_batch, tokenizer, embedding_model, max_length, device)

        # save to the memmap
        # Note: some experimenting indicated that indexing beyond the size of the memmap will reduce to
        # size of the memmap that was defined. Although the blog code makes use of this ... being loose
        # with the index makes me (perhaps unjustifiably?) nervous. In any case I decided to fix the
        # indices of the last batch (which may not be a full batch) so there are no broadcasting issues.
        nb = np.shape(embedded_batch)[0]
        fp[i * batch_size:i * batch_size + nb] = embedded_batch


# ----------------------------------------------------------------------------------------------------------------------
# Code for making a faiss index from
# ----------------------------------------------------------------------------------------------------------------------

def make_index(index_filename, num_rows=None, num_embed_dim=128, device="cpu"):
    """
    Makes a faiss index, either on cpu or gpu
    :param index_filename: (str): the filename of the memmap that stores the embeddings
    :param num_rows: (int): the number of rows (entries) in the embedding matrix
    :param num_embed_dim: the number of columns (dimensions) of the embedding matrix
    :param device: ('cpu' or 'cuda:x' the device upon which the index will live.
    :return: a faiss index that can be used for similarity search
    """

    # get the size of the embedding array if it is not known
    if num_rows is None:
        memmap_scratch = np.memmap(
            index_filename,
            dtype='float32', mode='r')
        num_rows = int(np.shape(memmap_scratch)[0] / num_embed_dim)

    # Make the memmap with the actual shape
    wiki40b_passage_reps = np.memmap(
        index_filename,
        dtype='float32', mode='r',
        shape=(num_rows, num_embed_dim)
    )

    # now make the index
    wiki40b_index_flat = faiss.IndexFlatIP(128)

    if device == 'cpu':
        print("Putting index on cpu.")
        wiki40b_index_flat.add(wiki40b_passage_reps)
        return wiki40b_index_flat
    elif device in ['cuda'] + ['cuda:' + str(i) for i in range(torch.cuda.device_count())]:
        print("Putting index on ", device)
        faiss_res = faiss.StandardGpuResources()
        wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
        wiki40b_gpu_index.add(wiki40b_passage_reps)
        return wiki40b_gpu_index
    else:
        raise ValueError("The device should be either 'cpu' or 'cuda:x'")


def query_index(question, embedding_model, tokenizer, wiki_dataset, kb_index,
                n_results=10, max_length=128, min_passage_length=20, device="cpu"):
    """

    :param question:
    :param embedding_model:
    :param tokenizer:
    :param wiki_dataset:
    :param kb_index:
    :param n_results:
    :param max_length:
    :param min_passage_length:
    :param device:
    :return:
    """

    embedding_model.to(device)

    # embed the question
    tokenized_question = tokenizer([question],
                                   max_length=max_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors='pt')

    with torch.no_grad():
        embedded_question = embedding_model.embed_questions(tokenized_question["input_ids"].to(device),
                                                            tokenized_question["attention_mask"].to(device))

    # now put on the cpu as numpy so we can do faiss. default, it should be on the cpu already
    embedded_question = embedded_question.cpu().numpy()

    # now query the index, using faiss. getting more than we need to make sure the text is long enough
    D, I = kb_index.search(embedded_question, 2 * n_results)

    # get the results of the query
    all_wikidata = [wiki_dataset[int(k)] for k in I[0]]
    all_passages = "<P> " + " <P> ".join([p["passage_text"] for p in all_wikidata])

    return all_passages


class QAKnowledgeBase:

    def __init__(self,
                 qar_model,
                 qar_tokenizer,
                 qa_s2s_model,
                 qa_s2s_tokenizer,
                 wiki_index,
                 wiki_dataset,
                 qar_device='cpu',
                 qa_s2s_device="cuda:0"):
        """
        Class that allows asking a question of the ELI5 model.
        :param qar_model: the RETRIEBERT model used for embedding
        :param qar_tokenizer: the tokenizer of the RETRIEBERT model
        :param qa_s2s_model: the BART sequence to sequence model
        :param qa_s2s_tokenizer: the tokenizer of the sequence to sequence model
        :param wiki_index: the faiss index of the wiki-snippets knowledge base
        :param wiki_dataset: the wiki-snippets dataset
        :param qar_device: the device (default 'cpu') upon which the RETRIEBERT model will run
        :param qa_s2s_device: the device (default 'cuda:0' upon which the BART sequence to sequence model will run.
        """
        self.qar_model = qar_model.to(qar_device)
        self.qa_s2s_model = qa_s2s_model.to(qa_s2s_device)

        self.qar_tokenizer = qar_tokenizer
        self.qa_s2s_tokenizer = qa_s2s_tokenizer

        self.wiki_index = wiki_index
        self.wiki_dataset = wiki_dataset

        self.qar_device = qar_device
        self.qa_s2s_device = qa_s2s_device

    def ask_question(self,
                     question,
                     num_answers=1,
                     num_beams=8,
                     min_answer_length=32,
                     max_answer_length=64,
                     do_sample=False,
                     temp=1.0,
                     top_p=None,
                     top_k=None,
                     max_input_length=1024):
        """
        Based upon the input question, queries the knowledge base and generates a response
        :param question: (str) the question being asked
        :param num_answers: (int) the number of unique answers returned, note, these will be identical unless do_sample=True
        :param num_beams: (int) number of beams in the beam search
        :param min_answer_length: # minimum number of tokens in the generated answer
        :param max_answer_length: # maximum number of tokens in the generated answer
        :param do_sample: # (bool): True if we are using top_k or top_p sampling, false otherwise
        :param temp: # beam search temp?
        :param top_p: # (float \in (0,1)) probability threshold for top_p sampling set to around 0.95 or so.
        :param top_k: # (int) number used for top k sampling
        :param max_input_length: (int) maximum number of tokens in input to BART model, includes both question and context
        :return: (list): a list of answers to the question asked
        """

        # make sure the number of beams is large enough for the number of answers asked for
        n_beams = num_answers if num_beams is None else max(num_beams, num_answers)

        # query the index
        doc = query_index(question, self.qar_model, self.qar_tokenizer,
                               self.wiki_dataset, self.wiki_index, device=self.qar_device)

        # concatenate question and support document into BART input
        question_doc = "question: {} context: {}".format(question, doc)

        # tokenize this with the BART tokenizer. Note we are truncating the total inputs at max_input_length
        s2s_inputs = self.qa_s2s_tokenizer([question_doc],
                                           max_length=max_input_length,
                                           padding="max_length",
                                           truncation=True,
                                           return_tensors='pt').to(self.qa_s2s_device)

        # now feed this into the BART sequence to sequence model and generate answers
        generated_ids = self.qa_s2s_model.generate(input_ids=s2s_inputs["input_ids"],
                                                   attention_mask=s2s_inputs["attention_mask"],
                                                   min_length=min_answer_length,
                                                   max_length=max_answer_length,
                                                   do_sample=do_sample,
                                                   early_stopping=True,
                                                   num_beams=1 if do_sample else n_beams,
                                                   temperature=temp,
                                                   top_k=top_k,
                                                   top_p=top_p,
                                                   eos_token_id=self.qa_s2s_tokenizer.eos_token_id,
                                                   no_repeat_ngram_size=3,
                                                   num_return_sequences=num_answers,
                                                   decoder_start_token_id=self.qa_s2s_tokenizer.bos_token_id)

        # and make the list of answers by decoding the generated ids
        answers = [self.qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]

        return answers

    def get_best_answers(self, question, num_candidates=5, top_p=0.95):
        """
        performs top p sampling and returns a list of answers sorted by how close they are in the embedding space
        to the question.
        :param question: (str) question asked
        :param num_candidates: number of unique answers
        :param top_p: token probability threshold for sampling
        :return: (list) list of answers sorted by embedding score (closeness in embedding space)
        """

        # ask the question, but with sampling turned on
        answers = self.ask_question(question, num_answers=num_candidates*5, top_p=top_p, do_sample=True)

        # tokenize and embed the questions
        tokenized_question = self.qar_tokenizer([question],
                                                max_length=128,
                                                padding="max_length",
                                                truncation=True,
                                                return_tensors='pt')

        with torch.no_grad():
            embedded_question = self.qar_model.embed_questions(tokenized_question["input_ids"].to(self.qar_device),
                                                               tokenized_question["attention_mask"].to(
                                                                   self.qar_device)).to('cpu').numpy()

        # embed the question
        embedded_answers = embed_passage_batch(answers,
                                               self.qar_tokenizer,
                                               self.qar_model,
                                               max_length=128,
                                               device=self.qar_device)

        scores = np.dot(embedded_answers, embedded_question.T).squeeze()

        sort_index = list(np.argsort(scores))
        sort_index.reverse()

        sorted_answers = [answers[i] for i in sort_index]

        return sorted_answers[0:num_candidates]
