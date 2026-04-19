# Indic Neural Machine Translation (NMT)

A PyTorch-based Sequence-to-Sequence (Seq2Seq) Neural Machine Translation model with Bahdanau Attention for translating between English and Hindi.

---

## Overview

This project implements a complete Neural Machine Translation pipeline using:

* Encoder-Decoder architecture
* Bidirectional LSTM Encoder
* Attention-based LSTM Decoder
* Custom tokenization and vocabulary handling
* Hugging Face dataset integration

The model is trained on the IIT Bombay English-Hindi parallel corpus.

---

## Model Architecture

### Encoder

* Bidirectional LSTM
* Embedding layer
* Hidden state projection to unify bidirectional outputs

### Attention

* Bahdanau (Additive) Attention
* Computes alignment between decoder state and encoder outputs

### Decoder

* LSTM with attention context
* Teacher forcing during training
* Token-by-token generation

---

## Project Structure

```
.
├── data.py        # Data preprocessing, vocabulary, dataloader
├── model.py       # Encoder, Attention, Decoder, Seq2Seq wrapper
├── train.py       # Training loop and dataset loading
├── README.md
```

---

## Dataset

This project uses the Hugging Face dataset:

* cfilt/iitb-english-hindi
* Parallel corpus of English-Hindi sentence pairs

---

## Installation

### 1. Create virtual environment (recommended)

```bash
python3.9 -m venv env
source env/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets nltk
```

---

## Training

Run:

```bash
python train.py
```

---

## Features

* Custom vocabulary building (`Lang` class)
* Dynamic padding using a custom collate function
* Teacher forcing mechanism
* Gradient clipping for stable training
* Automatic checkpoint saving

---

## Device Support

The code automatically selects the best available device:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
```

Ensure the correct PyTorch version is installed for GPU support.

---

## Training Details

| Parameter     | Value |
| ------------- | ----- |
| Hidden Size   | 256   |
| Embedding Dim | 256   |
| Batch Size    | 32    |
| Max Length    | 50    |
| Epochs        | 10    |
| Optimizer     | Adam  |

---

## Output

Model checkpoints are saved as:

```
nmt_checkpoint.pth
```

---

## Future Improvements

* Beam search decoding
* BLEU score evaluation
* Transformer-based architecture
* Multi-language support
* Subword tokenization (BPE)

---

## License

This project is intended for educational and research purposes.
