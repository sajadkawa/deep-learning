# Exercise 33 — Applications in NLP

## Section A — Multiple Choice

**1.** What property of word embeddings makes them more useful than one-hot vectors for NLP tasks?

- a) They are smaller in memory
- b) Semantically similar words have similar embedding vectors — similarity is encoded as geometry
- c) They can represent any number of vocabulary words without resizing
- d) They are computed without any training data

&nbsp;

**2.** In the Word2Vec skip-gram model, what is the model trained to predict?

- a) The centre word given the surrounding context words
- b) The surrounding context words given the centre word
- c) The next word in the sequence given all previous words
- d) The masked word given the full sentence context

&nbsp;

**3.** Perplexity of a language model is defined as:

- a) The fraction of tokens predicted correctly
- b) The exponentiated average negative log-probability per token
- c) The cross-entropy loss divided by sequence length
- d) The number of unique tokens in the vocabulary

&nbsp;

**4.** BERT uses masked language modelling as its pre-training task. What does this mean?

- a) The model predicts the next token given all previous tokens
- b) A fraction of tokens are hidden and the model predicts them using both left and right context
- c) The model is trained to classify sentences as positive or negative
- d) The model predicts whether two sentences are adjacent in the original text

&nbsp;

**5.** Which model architecture is better suited for text generation tasks and why?

- a) BERT, because it is bidirectional and sees the full context
- b) GPT, because it is trained autoregressively to predict the next token
- c) Word2Vec, because it produces the most accurate embeddings
- d) BERT, because it uses a larger vocabulary than GPT

&nbsp;

---

## Section B — Short Answer

**1.** Explain the difference between word tokenisation and subword tokenisation. Why do modern large language models use subword tokenisation?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Describe the skip-gram and CBOW variants of Word2Vec. What does each model receive as input and what does it predict?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** What are contextualised word embeddings and how do they differ from Word2Vec embeddings? Give an example where the distinction matters.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Explain how a pre-trained BERT model is fine-tuned for sentiment analysis. What is added to the model and what is trained?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A Word2Vec skip-gram model has vocabulary $\mathcal{V} = \{\text{deep, learning, neural, network, model}\}$ ($|\mathcal{V}| = 5$) and embedding dimension $d = 2$.

**Input embedding for "deep":** $v_{\text{deep}} = [0.5,\ -0.3]^T$

**Output vectors:**

$$u_{\text{learning}} = [0.4,\ 0.6]^T, \quad u_{\text{neural}} = [-0.2,\ 0.3]^T, \quad u_{\text{network}} = [0.1,\ -0.4]^T$$
$$u_{\text{model}} = [0.3,\ 0.2]^T, \quad u_{\text{deep}} = [-0.1,\ 0.5]^T$$

Use $e^{0.02} \approx 1.0202$, $e^{-0.19} \approx 0.8270$, $e^{0.17} \approx 1.1853$, $e^{0.09} \approx 1.0942$, $e^{-0.20} \approx 0.8187$, $\ln(0.2063) \approx -1.5773$.

**(a)** Compute the score $s_w = u_w^T v_{\text{deep}}$ for each word in the vocabulary.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute $e^{s_w}$ for each word and the sum $\sum_w e^{s_w}$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute $P(\text{learning} \mid \text{deep})$ using the softmax formula.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Compute the skip-gram loss $\mathcal{L} = -\log P(\text{learning} \mid \text{deep})$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** After one gradient step, $P(\text{learning} \mid \text{deep})$ increases to $0.35$. Compute the new loss. Has the model improved?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** Word2Vec embeddings are static: the embedding for "bank" is the same vector regardless of whether it appears in "river bank" or "bank account". Explain why this is a limitation for downstream NLP tasks and describe how transformer-based models address it.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A language model trained on a large English corpus achieves perplexity 25 on an English test set but perplexity 800 on a medical text test set. Explain what these numbers mean and why the gap exists. What would you do to improve performance on medical text?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
