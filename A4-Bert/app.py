import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads, self.d_model, self.d_k = n_heads, d_model, d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.W_O = nn.Linear(n_heads * d_k, d_model)
        self.scaled_dot_attn = ScaledDotProductAttention(d_k, device)
        self.layer_norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_O(context)
        return self.layer_norm(output + Q), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.device = device

    def forward(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output

class SiameseBERT(nn.Module):
    def __init__(self, bert_model, d_model):
        super(SiameseBERT, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(d_model * 3, 3)

    def mean_pool(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, p_ids, p_mask, h_ids, h_mask):
        # Segment IDs are all 0 for single sentence input in Siamese structure 
        p_seg = torch.zeros_like(p_ids)
        h_seg = torch.zeros_like(h_ids)
        
        u = self.bert(p_ids, p_seg)
        v = self.bert(h_ids, h_seg)
        
        u_pool = self.mean_pool(u, p_mask)
        v_pool = self.mean_pool(v, h_mask)
        
        combined = torch.cat([u_pool, v_pool, torch.abs(u_pool - v_pool)], dim=1)
        return self.classifier(combined)


from transformers import BertTokenizer

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    params = {
        'n_layers': 12,        
        'n_heads': 8,
        'd_model': 768, 
        'd_ff': 768 * 4, 
        'd_k': 96,             
        'n_segments': 2,
        'vocab_size': 207419,  
        'max_len': 128, 
        'device': device
    }
    
    base_bert = BERT(**params)
    model = SiameseBERT(base_bert, params['d_model'])
    
    try:
        state_dict = torch.load("sbert_best_model.pth", map_location=device)
        model.load_state_dict(state_dict, strict=False)
        st.success("Model and Tokenizer loaded!")
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        
    model.eval()
    return model, tokenizer, device

def predict_nli(premise, hypothesis, model, tokenizer, device):
    inputs_p = tokenizer(premise, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    inputs_h = tokenizer(hypothesis, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(
            inputs_p['input_ids'].to(device), 
            inputs_p['attention_mask'].to(device),
            inputs_h['input_ids'].to(device), 
            inputs_h['attention_mask'].to(device)
        )
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
    labels = ["Entailment", "Neutral", "Contradiction"]
    return labels[pred], probs[0][pred].item()


samples = {
    "Example 1": {
        "p": "A man is playing a guitar on stage.",
        "h": "The man is performing music."
    },
    "Example 2": {
        "p": "A soccer player is running with the ball.",
        "h": "The player is wearing red shoes."
    },
    "Example 3": {
        "p": "A black dog is running through the grass.",
        "h": "The dog is sleeping on the sofa."
    }
}

st.title("NLI Demo: Sentence BERT")
st.write("Determine the relationship between a premise and a hypothesis.")


model, tokenizer, device = load_model()

st.sidebar.header("Try a Sample")
sample_choice = st.sidebar.selectbox("Choose a pre-loaded example:", ["None"] + list(samples.keys()))

default_p = ""
default_h = ""

if sample_choice != "None":
    default_p = samples[sample_choice]["p"]
    default_h = samples[sample_choice]["h"]

col1, col2 = st.columns(2)
with col1:
    premise = st.text_area("Premise:", value=default_p, height=100, placeholder="Enter premise...")
with col2:
    hypothesis = st.text_area("Hypothesis:", value=default_h, height=100, placeholder="Enter hypothesis...")

if st.button("Predict Relationship", type="primary"):
    if premise and hypothesis:
        label, confidence = predict_nli(premise, hypothesis, model, tokenizer, device)
        colors = {"Entailment": "green", "Neutral": "orange", "Contradiction": "red"}
        color = colors.get(label, "blue")
        
        st.markdown(f"### Result: :{color}[{label}]")
        st.progress(confidence)
        st.write(f"Confidence Score: **{confidence:.2%}$**")
    else:
        st.warning("Please enter both a premise and a hypothesis.")