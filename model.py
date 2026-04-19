import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Processes the input sequence and produces fixed-size hidden states.
    """
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers=2, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # Bidirectional LSTM based on tf/nmt style
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True, batch_first=True)
        
        # Linear layer to convert bidirectional hidden state (2*hidden_size) to (hidden_size)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_length)
        embedded = self.embedding(input_seq)
        
        # output: (batch_size, seq_length, 2*hidden_size)
        # hidden, cell: (2*num_layers, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional states
        batch_size = hidden.shape[1]
        
        # Reshape to (num_layers, 2, batch_size, hidden_size)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate forward and backward directions for the final states
        hidden_cat = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell_cat = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        # Transform back to hidden_size
        hidden_final = torch.tanh(self.fc_hidden(hidden_cat))
        cell_final = torch.tanh(self.fc_cell(cell_cat))
        
        return output, (hidden_final, cell_final)


class BahdanauAttention(nn.Module):
    """
    Additive Attention mechanism used in the original seq2seq architectures.
    Calculates attention scores between current decoder state and all encoder outputs.
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)  # *2 because encoder output is bidirectional
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1)
        
        # Calculate alignment scores: (batch_size, seq_length, 1)
        scores = self.Va(torch.tanh(self.Wa(decoder_hidden_expanded) + self.Ua(encoder_outputs)))
        
        # Softmax over seq_length: (batch_size, seq_length, 1)
        attn_weights = F.softmax(scores, dim=1)
        
        # Context vector: (batch_size, 1, 2*hidden_size)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        
        return context, attn_weights


class AttnDecoderRNN(nn.Module):
    """
    LSTM Decoder with Attention.
    Generates the output sequence one token at a time.
    """
    def __init__(self, hidden_size, output_size, embedding_dim, num_layers=2, dropout=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.attention = BahdanauAttention(hidden_size)
        
        # Input to LSTM combining embedding and attention context vector
        self.lstm = nn.LSTM(embedding_dim + hidden_size * 2, hidden_size, 
                            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (batch_size, 1)
        embedded = self.dropout(self.embedding(input_token))
        
        # Use top layer's hidden state for attention
        decoder_hidden = hidden[-1]
        
        # Calculate attention context
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs)
        
        # Concatenate embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        # Prediction
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """
    Wrapper mapping the Encoder and Decoder dynamically.
    """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # outputs tensor to store predictions
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # encode the source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # First input to the decoder is the SOS token.
        input_token = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs
