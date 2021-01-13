# emcdr
Cross-Domain Recommendation: An Embedding and Mapping Approach

## Directory Structure
 - model depending > data depending
```
emcdr/
¦-- main.py
¦-- util.py
¦-- .gitignore
¦
+-- Latent_Factor_Modeling/
¦   +-- bpr.py
¦   +-- doc2vec.py
¦   +-- lda.py
¦
+-- Latent_Space_Modeling/
¦   +-- mlp.py 
¦
+-- model/
¦       +-- bpr/
¦       +-- doc2vec/
¦       ¦    +-- doc2vec_{PARAMETERS}_{source / target}_{trained / mapped}
¦       +-- lda/
¦            +-- lda_{PARAMETERS}_{source / target}_{trained / mapped}
¦ 
+-- vector/
¦   +-- users/
¦       +-- bpr/
¦       ¦    +-- bpr_{PARAMETERS}_{source / target}_{trained / mapped}
¦       +-- doc2vec/
¦       ¦    +-- doc2vec_{PARAMETERS}_{source / target}_{trained / mapped}
¦       +-- lda/
¦            +-- lda_{PARAMETERS}_{source / target}_{trained / mapped}
+-- data/
```
