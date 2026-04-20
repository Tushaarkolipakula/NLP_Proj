import argparse
import torch
from model import EncoderRNN, AttnDecoderRNN, Seq2Seq
from data import Lang, normalize_string
from evaluate import evaluate_sentence

class Translator:
    """
    Wrapper class to load the model checkpoint and expose a simple translation API.
    """
    def __init__(self, checkpoint_path, input_lang, output_lang, hidden_size=256, embedding_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_lang = input_lang
        self.output_lang = output_lang
        
        # Initialize model architecture matching training structures
        encoder = EncoderRNN(input_lang.n_words, hidden_size, embedding_dim).to(self.device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, embedding_dim).to(self.device)
        
        self.model = Seq2Seq(encoder, decoder, self.device).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"Successfully loaded model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_path}. Ensure it exists.")
            print(f"Exception: {e}")

    def translate(self, text):
        norm_text = normalize_string(text)
        try:
            output_words = evaluate_sentence(self.model, norm_text, self.input_lang, self.output_lang)
            # Post-process, strip EOS if present
            if '<EOS>' in output_words:
                output_words.remove('<EOS>')
            return ' '.join(output_words)
        except KeyError as e:
            return f"[Error] Unknown word in input: {e}"
        except Exception as e:
            return f"[Error] Translation failed: {e}"

if __name__ == "__main__":
    import pickle
    parser = argparse.ArgumentParser(description="Indic NMT Translator CLI")
    parser.add_argument("--checkpoint", type=str, default="nmt_checkpoint.pth", help="Path to the model checkpoint.")
    parser.add_argument("--vocab", type=str, default="vocab.pkl", help="Path to the vocabulary pickle file.")
    
    args = parser.parse_args()
    
    print(f"Loading vocabulary from {args.vocab}...")
    with open(args.vocab, 'rb') as f:
        input_lang, output_lang = pickle.load(f)
        
    print("Initialize translator...")
    translator = Translator(args.checkpoint, input_lang, output_lang)
    
    print("\nNMT Interactive Interface. Type 'q' to quit.")
    while True:
        text = input("English > ")
        if text.strip().lower() == 'q':
            break
        
        translation = translator.translate(text)
        print(f"Hindi Validation > {translation}")
