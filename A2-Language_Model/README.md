# A2: Language Model - Biblical Language Model Web App

This application demonstrates two different LSTM architectures trained on the King James Bible to perform next-word prediction and text generation.

## Project Structure
- `bible.ipynb`: Notebook showing the working of the assignment. Training code along with result interpretation is presented in the notebook. 
- `app.py`: Streamlit application containing model architectures and UI logic.
- `best_basic_lstm.pt`: Trained weights for the standard LSTM - Baseline.
- `best_attn_lstm.pt`: Trained weights for the Causal Masked Attention LSTM - ImprovedAttention.
- `vocab.pkl`: The pickled vocabulary mapping used during training.
- `requirements.txt`: Required packages to run the app
- `A2-demo.mov`: Demo video

## How it Works
The application uses **Streamlit** to provide a web interface. 
1. **Input Handling**: The user provides a text prompt which is tokenized using NLTK's `basic_english` tokenizer.
2. **Inference**: The indices are passed to the selected model. The model performs a forward pass to produce logits.
3. **Sampling**: We apply a **Softmax with Temperature** (set to 0.85) to the last predicted token to choose the next word. This process repeats until the desired length is reached.
4. **Output**: The generated string is decoded using the vocabulary and displayed on the UI.

## Setup Instructions
0. cd `A2-Language_Model`
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure all `.pt` and `.pkl` files are in the same directory as `app.py`.
3. Run the app: `streamlit run app.py`


## Video Demo
<video src="A2-demo.mp4" width="100%" controls></video>