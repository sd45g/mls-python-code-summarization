# Models Directory

Place your trained model files here:

- `best.pt` - The best model checkpoint (lowest validation loss)
- `last.pt` - The most recent checkpoint (optional, for resuming training)

## ⚠️ Model Not Included in Repository

The trained model file is too large for GitHub (>300MB).

**Download from Google Drive:**
[Add your public Google Drive link here after training]

## Setup Instructions

1. Download `best.pt` from the Google Drive link above
2. Place it in this `models/` folder
3. Run inference: `python summarize.py --input "def add(a, b): return a + b"`

## Expected File Size

The trained model is approximately 300+ MB.
