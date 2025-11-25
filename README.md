# Vector-Quantizer-Assignment

CLI tool to experiment with vector quantizer-based image compression.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python vector_quantizer.py
```

The program shows a menu:

1. **Compression**
   - Provide the image path.
   - Choose whether to specify the block dimensions (height/width) or the number of blocks per dimension. Padding is applied automatically when the image dimensions are not divisible by the block size.
   - Enter the desired codebook size (e.g., 16, 32, 64). Larger values usually improve quality but reduce compression.
   - Outputs:
     - Codebook file (`*_codebook.npy`)
     - Compressed representation (`*_compressed.npz`)
     - Preview image produced from the quantized data
     - Estimated compression ratio shown in the terminal
2. **Decompression**
   - Provide the path to a previously generated `*_compressed.npz` file.
   - The program recreates the image using the stored indices and codebook, writing `outputs/decompressed_<timestamp>.png`.
3. **Exit**

All generated artifacts are stored under the `outputs/` directory. The compressed `.npz` bundle contains everything needed for decompression (original shape, block size, padding, and a pointer to the saved codebook).
