# EntityLinker

Curation of methods and programs to link records with an emphasis on names and addresses.
To use library:
```
git clone https://github.com/jdm365/EntityLinker.git
cd EntityLinker
make
```

To use with environment.yml:
```
git clone https://github.com/jdm365/EntityLinker.git
cd EntityLinker
conda env create -f environment.yml
conda activate EntityLinkerEnv
make
```

![Entity Linker Package Diagram](utils/EntityLinkerPackageDiagram.png)

Each of the three categories of methods have various utility functions located 
in the `lib` packages. Main utility functions common to all libs are the `dedup`
functions. Some of the libs also offer similarity search functions, but often times
the recommended approach is to simply use the underlying libs alone 
(i.e. faiss, datasketch lsh, pynndescent, etc.)

Evaluation of each of the methods against a test dataset of companies can be seen
in the eval directories. This will run the dedup/sim_search algorithm against corrupted
data (corrupted by `utils/fuzzify.py`) and report the recall scores @5. None of the methods
here showed nearly good enough standalone performance (for deduplication) to be used alone
so we avoided using precision as a metric and rather view these algorithms as 
"candidate generation methods". Therefore a high recall and reasonable runtime are very
valuable but excellent precision is not as important.


We offer three methods in this repo chiefly supported by the libraries datasketch (for lsh),
sklearn (for tfidf and kd_tree knn), HuggingFace (for the char_bert, mainly Byt5, model), 
and faiss (for rapid similarity search).

Method 1: Locality Sensitive Hashing
Locality sentive hashing is done by hashing shingles of a string with N different
hashing functions. The shingles are then hashed into buckets. The buckets are then
compared to find the most similar strings using the jaccard similarity metric.


Method 2: TFIDF Nearest Neighbor Search
TFIDF vectorizes text by getting the counts of each token which occured
in a document, representing this as a `vocab_size` dimensional vector.
The TFIDF vector is then normalized by the number of tokens in the document.


Method 3: Pretrained Charachter Level LLM Embedding Nearest Neighbor Search
Byt5 is a pretrained language model from Google which uses a charachter level tokenizer.
The model is part of the T5 family of models. 



FUNCS:
LSH
dedupe_lsh(items, model, k=5)


TFIDF
dedupe_knn(items, k=5)


BYT5
HuggingFaceByt5Wrapper()
	forward(X: str)
	get_embedding(X: str)

dedupe_byt5(items, model, k=5)
