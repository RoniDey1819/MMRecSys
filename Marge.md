# Merge — Text + Image Embeddings

Joins the per-interaction output files from the BERT and ViT embedding notebooks
into a single file ready for sentiment prototype learning.

### Inputs

- `interactions_text_embeddings.jsonl` — output of BERT notebook
- `interactions_image_embeddings.jsonl` — output of ViT notebook

### Output

- `interactions_multimodal.jsonl` — one record per `(user_id, asin)` pair
  where **both** a text embedding and an image embedding exist.

### Why both must be present

The prototype learning forward pass requires `vi` and `tui` simultaneously
to compute the consistency loss `|| z_v - z_t ||` and to backpropagate
through both prototype attention weights in the same step.
Interactions missing either modality cannot be used for training and are
excluded here rather than silently zero-padded inside the model.

### Pipeline position

```
preprocess.ipynb
       |
       |----------------------------------+
       v                                  v
BERT_embedding.ipynb            ViT_embedding.ipynb
       |                                  |
       v                                  v
text_embeddings.jsonl       image_embeddings.jsonl
       |                                  |
       +-----------------+----------------+
                         v
                   merge.ipynb  <-- YOU ARE HERE
                         |
                         v
          interactions_multimodal.jsonl
```
