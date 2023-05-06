# EntityLinker
from scipy.optimize import differential_evolution


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
