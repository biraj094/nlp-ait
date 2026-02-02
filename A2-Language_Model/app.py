import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
import pickle
import math


class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        return self.fc(out)

class AttentionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lin_Q = nn.Linear(hidden_dim, hidden_dim)
        self.lin_K = nn.Linear(hidden_dim, hidden_dim)
        self.lin_V = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def self_attention(self, x):
        x = self.layer_norm(x)
        Q, K, V = self.lin_Q(x), self.lin_K(x), self.lin_V(x)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.size(-1))
        mask = torch.triu(torch.ones(scores.size(-1), scores.size(-1), device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(self.dropout(attn), V)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.self_attention(out)
        return self.fc(out)


@st.cache_resource
def load_assets():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    tokenizer = get_tokenizer('basic_english')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Basic Model - config from training
    basic_model = BasicLSTM(len(vocab), 256, 512, 2, 0.4).to(device)
    basic_model.load_state_dict(torch.load('best_basic_lstm.pt', map_location=device))
    basic_model.eval()

    # Load Attention Model - config from training
    attn_model = AttentionLSTM(len(vocab), 256, 256, 2, 0.3).to(device)
    attn_model.load_state_dict(torch.load('best_attn_lstm.pt', map_location=device))
    attn_model.eval()
    
    return vocab, tokenizer, basic_model, attn_model, device

vocab, tokenizer, basic_model, attn_model, device = load_assets()


def generate(model, prompt, max_words, temp=0.85):
    tokens = tokenizer(prompt)
    indices = [vocab[w] for w in tokens]
    for _ in range(max_words):
        input_tensor = torch.LongTensor([indices]).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output[0, -1, :] / temp, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)
            if next_idx == vocab['<eos>']: break
    return ' '.join(vocab.lookup_tokens(indices))


st.set_page_config(page_title="Bible AI Generator")
st.title("📜 Biblical Language Model")
st.markdown("Compare a standard LSTM against a Causal Masked Attention LSTM.")

prompt = st.text_input("Enter a prompt:", value="the lord is")
max_len = st.slider("Length of generation", 10, 100, 30)

if st.button("Generate Text"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic LSTM")
        with st.spinner('Generating...'):
            res_basic = generate(basic_model, prompt, max_len)
            st.write(f"*{res_basic}*")
            
    with col2:
        st.subheader("Attention LSTM")
        with st.spinner('Generating...'):
            res_attn = generate(attn_model, prompt, max_len)
            st.write(f"*{res_attn}*")