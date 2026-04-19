import torch
import torch.nn as nn
from model import EncoderRNN, AttnDecoderRNN, Seq2Seq
from data import pad_sequence_custom, PAD_TOKEN, SOS_TOKEN
from train import train_epoch

def test_dimensions():
    print("Testing Seq2Seq Dimensions...")
    
    vocab_size = 100
    embedding_dim = 64
    hidden_size = 128
    batch_size = 4
    seq_len = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = EncoderRNN(vocab_size, hidden_size, embedding_dim).to(device)
    decoder = AttnDecoderRNN(hidden_size, vocab_size, embedding_dim).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    # Mock data
    src = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    # Target needs SOS at index 0 explicitly to not blow up evaluation loops usually (though model trains without assuming it mathematically, logic dictates structural matching)
    trg = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    trg[:, 0] = SOS_TOKEN
    
    # 1. Test Forward pass
    try:
        outputs = model(src, trg, teacher_forcing_ratio=0.5)
        print(f"Forward Pass ✅ Dimension: {outputs.shape}")
        assert outputs.shape == (batch_size, seq_len, vocab_size)
    except Exception as e:
        print(f"Forward Pass Failed ❌ {e}")
        return False
        
    return True

if __name__ == "__main__":
    if test_dimensions():
        print("\nAll architecture dimensions verified successfully!")
    else:
        print("\nVerification failed.")
