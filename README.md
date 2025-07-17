# LLM Preference Classification with DeBERTa + Logistic Regression

##  Project Overview

This project predicts which chatbot response a human user prefers in a head-to-head comparison between two LLM-generated outputs. The model is evaluated on its ability to match human preference judgments.


---

## Methodology

The approach uses a frozen **DeBERTaV3** model to extract contextual embeddings for each prompt-response pair. These embeddings are used as input features for a **logistic regression classifier**.

The data preprocessing and tokenization process builds on [this Kaggle notebook](https://www.kaggle.com/code/gusthema/lmsys-kerasnlp-starter/notebook), but differs in several key ways:

- No classification head or softmax layer is used.
- Instead, contextual embeddings from DeBERTa are directly passed to `scikit-learn`'s logistic regression model.
- Each pair is converted into a string format:
  ```
  Prompt: What is the difference between a frog and a toad?
  Response: A frog is...
  ```

## Embedding Strategy

Initial experiments used only concatenation of the CLS token embeddings for `emb_a` and `emb_b`. However, this setup performed poorly on the **"Tie"** class (recall and F1 ≈ 0.33).

### Feature Improvements

The feature vector was extended to include additional embedding interactions:

```python
X = np.concatenate([
  emb_a,              
  emb_b,              
  emb_a - emb_b,      # Directional difference
  emb_a * emb_b       # Element-wise interaction
], axis=1)

```
Logistic Regression setup `class_weight='balanced'` to adjusts the underrepresented "Tie" class.

## Final Results

After applying the embedding modifications and tuning the logistic regression model:

- **Macro F1-score** improved across all classes.
- **"Tie" class** showed improvement in recall and F1-score.

Despite the progress, performance remains constrained by:
- The linear nature of logistic regression, which may not capture complex relationships between embeddings.
- CLS token embeddings extracted from a frozen DeBERTa model, which may not be optimal for the preference classification task without fine-tuning.

---

## Future Work

- May replace logistic regression with more powerful models like XGBoost or a lightweight neural network to better capture nonlinear relationships.
- Fine-tune the DeBERTa encoder.





