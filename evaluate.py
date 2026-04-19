import torch
from data import indexes_from_sentence, SOS_TOKEN, EOS_TOKEN
from nltk.translate.bleu_score import corpus_bleu
import nltk

# Ensure tokenization dependencies (if needed later)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def evaluate_sentence(model, sentence, input_lang, output_lang, max_length=50):
    """
    Greedy decoding iteration for evaluating a single sentence.
    """
    model.eval()
    with torch.no_grad():
        indexes = indexes_from_sentence(input_lang, sentence)
        indexes.append(EOS_TOKEN)
        input_tensor = torch.tensor(indexes, dtype=torch.long, device=model.device).view(1, -1)
        
        # Get encoder outputs
        encoder_outputs, (hidden, cell) = model.encoder(input_tensor)
        
        # Start of sequence
        decoder_input = torch.tensor([[SOS_TOKEN]], device=model.device)
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for t in range(max_length):
            output, hidden, cell, attn_weights = model.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            # Using greedy decoding (highest probability token)
            topv, topi = output.topk(1)
            
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            
            # Next input is the current predicted output
            decoder_input = topi.detach()
            
        return decoded_words

def evaluate_bleu(model, pairs, input_lang, output_lang):
    """
    Rubric Generator to evaluate Machine Translation Quality using BLEU.
    """
    references = []
    candidates = []
    
    for i, (src, trg) in enumerate(pairs):
        # Trg is the reference sentence
        trg_words = trg.split(' ')
        references.append([trg_words])  # corpus_bleu expects a list of references per candidate
        
        # Candidate is the model prediction
        output_words = evaluate_sentence(model, src, input_lang, output_lang)
        if '<EOS>' in output_words:
            output_words.remove('<EOS>')
        
        candidates.append(output_words)
        
        if i % 100 == 0:
            print(f"Evaluated {i}/{len(pairs)} pairs...")
            
    # Calculate overarching BLEU score
    # We use 4-gram BLEU scores with smoothing as the primary rubric for NMT
    from nltk.translate.bleu_score import SmoothingFunction
    smoothie = SmoothingFunction().method4
    
    score = corpus_bleu(references, candidates, smoothing_function=smoothie)
    print(f"===========================================================")
    print(f"RUBRIC SCORE -> Corpus BLEU score over {len(pairs)} pairs: {score * 100:.2f}")
    print(f"===========================================================")
    
    return score
