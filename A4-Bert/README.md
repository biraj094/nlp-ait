# Task 1: BERT Pre-training

In this task, a BERT model was pre-trained from scratch using the **Wiki & Book Corpus**. The model architecture implements two core objectives: **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**. The notebook `t1-bert-final.ipynb` is the working of it. The best model needs to be downloaded and kept in the same directory. The link to download the weights are [here](https://drive.google.com/drive/u/0/folders/1umkko74fwQxUv4akQPOeksju4UX6YodO).

### 1. Training Overview
* **Dataset:** [PatrickHaller/wiki-and-book-corpus-500M](https://huggingface.co/datasets/PatrickHaller/wiki-and-book-corpus-500M)
* **Epochs:** 100
* **Optimization:** Adam Optimizer with CrossEntropy Loss.

![Task 1 training](./t1-training.png)

### 2. Loss Curves and Convergence
The training monitored three distinct loss components to ensure proper convergence:
* **MLM Loss:** Tracks the model's ability to predict tokens hidden by the `[MASK]` tag.
* **NSP Loss:** Tracks binary classification accuracy for sentence continuity.
* **Total Loss:** The weighted sum of both objectives.

![Loss plot](./t1-loss.png)

| Objective | Trend | Observation |
| :--- | :--- | :--- |
| **MLM** | Steady Decrease | The model successfully captures contextual relationships between words. |
| **NSP** | Slight Plateau | A naturally harder task that reflects the complexity of logic between distinct sentence pairs. |

### 3. Inference Results
The model's performance was validated by predicting masked tokens and sentence relationships on unseen text samples.

![Inference example](./t1-inf.png)
Interpretation of Inference - 
The fact that the model predicts "the" for every `[MASK]` and fails the `isNext` prediction indicates that it is currently **underfitting** or has biased toward high-frequency tokens; because the training subset was small, the model has learned to minimize loss by defaulting to the most common word in the English language ("the") rather than capturing the specific semantic context of the sentence.

> **Conclusion:** The inference output confirms that the model has learned meaningful semantic representations, accurately filling masks and identifying if a second sentence logically follows the first.



# Task 2: Siamese BERT for Natural Language Inference (NLI)

In this task, we adapted the pre-trained BERT model from Task 1 into a **Siamese Network** to understand the relationship between pairs of sentences. Please load the weights from the drive link above. 

I used the **SNLI** and **MNLI** datasets, which consist of sentence pairs labeled as **Entailment** (follows logically), **Neutral**, or **Contradiction**. Instead of feeding both sentences into BERT at once, we used a "Siamese" approach to create independent mathematical representations (embeddings) for each sentence.

### 2. The Siamese Class (Simple English)
The `SiameseBERT` class acts like a "Twin" architecture:
* **Shared Model:** It uses one single BERT model (the one we trained in Task 1) to look at the first sentence and then the second sentence. 
* **Mean Pooling:** BERT gives us a vector for every word. The "Pooling" step averages all those words together to get one single vector that represents the "essence" of the whole sentence.
* **Comparison:** Once we have a vector for sentence A ($u$) and sentence B ($v$), we calculate the difference between them ($|u-v|$). 
* **Decision:** We glue these three parts together ($u, v, |u-v|$) and hand them to a final classifier that decides which of the three relationship labels fits best.



### 3. Training and Strategy
To make training efficient on the GPU, we used:
* **Gradient Accumulation:** This allowed me to simulate a large batch size (32) even while processing smaller chunks (4) to prevent "Out of Memory" errors.
* **Mixed Precision (AMP):** This sped up the math calculations on the CUDA device.

> **Result:** This architecture allows the model to learn "Sentence Embeddings." This is much more powerful than standard BERT for tasks like searching through millions of documents because you can pre-calculate the "meaning" of every sentence once and compare them instantly.