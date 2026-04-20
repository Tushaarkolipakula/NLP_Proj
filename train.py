import torch
import torch.nn as nn
from model import Seq2Seq
from data import PAD_TOKEN, EOS_TOKEN
import time
import math

def train_epoch(model, dataloader, optimizer, criterion, clip=1.0):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(model.device), trg.to(model.device)
        
        optimizer.zero_grad()
        
        # trg is structurally identical to src, outputs gives probabilities over vocab
        output = model(src, trg, teacher_forcing_ratio=0.5)
        
        # output: (batch_size, trg_len, output_dim)
        # trg: (batch_size, trg_len)
        output_dim = output.shape[-1]
        
        # Flatten the outputs for the loss function, skipping the SOS token input
        # trg[:, 1:] because the model predicts starting from the 1st actual word.
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Batch {i}/{len(dataloader)} Loss: {loss.item():.4f}")
            
        # Intermediate saves to prevent progress loss
        if i % 2500 == 0 and i > 0:
            print(f"Intermediate Checkpoint saved at batch {i}!")
            torch.save(model.state_dict(), "nmt_checkpoint.pth")
            
    return epoch_loss / len(dataloader)


def train_model(model, dataloader, optimizer, num_epochs=5, save_path="nmt_checkpoint.pth", patience=2):
    """
    Main training harness with early stopping.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Timing utilities
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return f"{m}m {s:.0f}s"
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, dataloader, optimizer, criterion)
        
        print(f"Epoch: {epoch+1:02} | Time Elapsed: {as_minutes(time.time() - start_time)}")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        
        # Early stopping & save condition
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            print(f"Saving checkpoint to {save_path}...")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement in loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
            
    print("Training finished!")

if __name__ == "__main__":
    from datasets import load_dataset
    import torch.optim as optim
    from data import Lang, normalize_string, get_dataloader
    from model import EncoderRNN, AttnDecoderRNN, Seq2Seq
    
    print("Loading dataset...")
    ds = load_dataset("cfilt/iitb-english-hindi")
    
    print("Preparing data...")
    pairs = []
    
    # Iterate through the train split to prepare pairs
    # Note: You may want to slice the dataset (e.g., ds['train'].select(range(10000))) for faster local training
    for item in ds['train']:
        en_text = normalize_string(item['translation']['en'])
        hi_text = normalize_string(item['translation']['hi'])
        pairs.append([en_text, hi_text])
        
    input_lang = Lang("en")
    output_lang = Lang("hi")
    
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
        
    print(f"Original Input Vocab: {input_lang.n_words} words")
    print(f"Original Output Vocab: {output_lang.n_words} words")
    
    # Trim rare words occurring fewer than 2 times to prevent Softmax layer blowup
    input_lang.trim(2)
    output_lang.trim(2)
        
    print(f"Trimmed Input Vocab: {input_lang.n_words} words")
    print(f"Trimmed Output Vocab: {output_lang.n_words} words")
    print(f"Total pairs: {len(pairs)}")
    
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump((input_lang, output_lang), f)
    print("Saved vocab.pkl")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    HIDDEN_SIZE = 256
    EMBEDDING_DIM = 256
    BATCH_SIZE = 128
    MAX_LEN = 50
    NUM_EPOCHS = 5
    
    print("Creating dataloader...")
    dataloader = get_dataloader(pairs, input_lang, output_lang, batch_size=BATCH_SIZE, max_len=MAX_LEN)
    
    print("Initializing model...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE, EMBEDDING_DIM).to(device)
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, EMBEDDING_DIM).to(device)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    train_model(model, dataloader, optimizer, num_epochs=NUM_EPOCHS)
