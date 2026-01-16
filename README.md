# Python Code Summarization

Automatic Python code summarization using Transformer encoder-decoder networks.

**Training Environment:** Google Colab (GPU)

---

## 1. Setup & Installation

### For Google Colab (Training)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/sd45g/mls-python-code-summarization.git
%cd mls-python-code-summarization
!git checkout fix/improve-model-training

# Install dependencies
!pip install -r requirements.txt
```

### For Local Machine (Inference Only)

```bash
pip install -r requirements.txt
```

Dependencies:

- `torch` - Deep learning framework
- `tokenizers` - BPE tokenizer
- `datasets` - Hugging Face data loading
- `tqdm` - Progress bars
- `rouge-score` - ROUGE evaluation
- `nltk` - BLEU and METEOR evaluation
- `numpy` - Numerical operations

---

## 2. Execution Commands

### Training (on Google Colab)

**Step 1: Preprocess Data** (first time only)

```bash
python scripts/preprocess_data.py
```

**Step 2: Train Tokenizer** (first time only)

```bash
python scripts/train_tokenizer.py
```

**Step 3: Train Model**

```bash
python scripts/train.py
```

> **Note:** Steps 1 and 2 only need to be run once. The preprocessed data and tokenizer are saved and reused.

### Evaluation

```bash
python scripts/evaluate.py
```

This will compute:

- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU
- METEOR
- Perplexity

### Inference (Summarize Code)

```bash
python summarize.py --input "def my_function(x, y): return x+y"
```

Or run in interactive mode:

```bash
python summarize.py
```

---

## 3. Model Location

The trained model is saved to Google Drive during training:

```
/content/drive/MyDrive/mls-python-code-summarization/models/best.pt
```

**Google Drive Link (for viewing/backup):**
https://drive.google.com/drive/folders/1DENgd2_-jh1-wvwgA3arKGWATPBEAI_I?usp=sharing

The model is automatically loaded from Google Drive when running `summarize.py` or `evaluate.py` on Colab.

---

## 4. Project Structure

```
├── scripts/
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── inference.py       # Core inference functions
│   ├── preprocess_data.py # Data preprocessing
│   └── train_tokenizer.py # Tokenizer training
├── src/
│   ├── transformer_model.py      # Transformer architecture
│   ├── train_utils_transformer.py # Training utilities
│   └── data.py                   # Dataset and collator
├── summarize.py           # CLI for code summarization
├── requirements.txt       # Dependencies
└── REPORT.md             # Full project report
```

---

## 5. Authors

- Sarah Ahmed Abushaala
- Mariia Al-Kafri
- Ahmad Sohail Najib

**GitHub Repository**: https://github.com/sd45g/mls-python-code-summarization
