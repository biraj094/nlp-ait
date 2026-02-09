import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchtext.data.utils import get_tokenizer
from indicnlp.tokenize import indic_tokenize

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim, method="additive"):
        super().__init__()
        self.method = method
        if method == "additive":
            self.W = nn.Linear(hid_dim, hid_dim)
            self.U = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)
        else: # general
            self.W = nn.Linear(hid_dim, hid_dim * 2)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[0]
        s = hidden.unsqueeze(1).repeat(1, src_len, 1)
        projected_encoder = encoder_outputs.permute(1, 0, 2)
        if self.method == "additive":
            energy = torch.tanh(self.W(s) + self.U(projected_encoder))
            energy = self.v(energy).squeeze(2)
        else:
            s = self.W(hidden).unsqueeze(2)
            energy = torch.bmm(projected_encoder, s).squeeze(2)
        energy = energy.masked_fill(mask, -1e4)
        return F.softmax(energy, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs, mask)
        a_unsqueezed = a.unsqueeze(1)
        enc_perm = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a_unsqueezed, enc_perm).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0)
        return mask

@st.cache_resource
def load_assets(model_type):
    with open('vocab_transform.pkl', 'rb') as f:
        vocab = pickle.load(f)
    device = torch.device('cpu')
    HID_DIM = 512
    
    filename = "Additive_Attention_best.pt" if model_type == "Additive" else "General_Attention_best.pt"
    method = "additive" if model_type == "Additive" else "general"
    
    attn = Attention(HID_DIM, method=method)
    enc = Encoder(len(vocab['en']), 256, HID_DIM, 0.5)
    dec = Decoder(len(vocab['ne']), 256, HID_DIM, 0.5, attn)
    model = Seq2Seq(enc, dec, 1, device).to(device)
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, get_tokenizer('basic_english'), device

st.set_page_config(page_title="MT English/Nepali", layout="centered")
st.title("Machine Translation English/Nepali")

model_choice = st.radio("अनुवाद मोड (Attention Mode):", ("Additive", "General"), horizontal=True)
model, vocab, en_tok, device = load_assets(model_choice)

samples = [
    "Where is the market?",
    "Invest in these foreign funds.",
    "Investors are bullish on the stock market."
]

st.write("नमूना वाक्यहरू (Sample Sentences):")
cols = st.columns(3)
if 'input_val' not in st.session_state:
    st.session_state.input_val = ""


for i, sample in enumerate(samples):
    if cols[i].button(sample):
        st.session_state.input_val = sample


input_text = st.text_area("अङ्ग्रेजी यहाँ लेख्नुहोस्:", value=st.session_state.input_val, height=100)

if st.button("अनुवाद गर्नुहोस् (Translate)"):
    if input_text.strip():
        with st.spinner('अनुवाद हुँदैछ...'):
            tokens = en_tok(input_text.lower())
            indices = [vocab['en']['<sos>']] + [vocab['en'][t] for t in tokens] + [vocab['en']['<eos>']]
            src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)
            src_len = torch.LongTensor([len(indices)])

            with torch.no_grad():
                encoder_outputs, hidden = model.encoder(src_tensor, src_len)
                mask = model.create_mask(src_tensor)
                trg_idx = [vocab['ne']['<sos>']]
                for _ in range(50):
                    trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
                    output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
                    pred = output.argmax(1).item()
                    trg_idx.append(pred)
                    if pred == vocab['ne']['<eos>']: break
            
            result = " ".join([vocab['ne'].get_itos()[i] for i in trg_idx if i not in [1, 2, 3]])
            st.success("नेपाली अनुवाद:")
            st.subheader(result)