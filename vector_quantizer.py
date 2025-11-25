import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def prompt_menu() -> str:
    print("\nVector Quantizer")
    print("1) Compress Image")
    print("2) Decompress Image")
    print("3) Exit")
    return input("Enter choice: ").strip()


def read_positive_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a positive integer.")


def load_image(path: str) -> Tuple[np.ndarray, str]:
    image = Image.open(path)
    image = image.convert("RGB")
    return np.array(image, dtype=np.uint8), image.mode


def pad_image(image: np.ndarray, block_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    block_h, block_w = block_size
    h, w, _ = image.shape
    pad_h = (block_h - (h % block_h)) % block_h
    pad_w = (block_w - (w % block_w)) % block_w
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (pad_h, pad_w)


def split_blocks(image: np.ndarray, block_size: Tuple[int, int]) -> np.ndarray:
    block_h, block_w = block_size
    h, w, c = image.shape
    blocks = []
    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            block = image[y:y + block_h, x:x + block_w, :]
            blocks.append(block.flatten())
    return np.vstack(blocks)


def kmeans(data: np.ndarray, k: int, max_iter: int = 40, tol: float = 1e-4, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(data), size=k, replace=False)
    centroids = data[indices].astype(np.float32)
    labels = np.zeros(len(data), dtype=np.int32)

    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            members = data[labels == i]
            if len(members) > 0:
                centroid = members.mean(axis=0)
            else:
                centroid = data[rng.integers(len(data))]
            centroids[i] = centroid

        shift = np.linalg.norm(centroids - prev_centroids)
        if shift < tol:
            break

    return centroids, labels


def reconstruct_image(codebook: np.ndarray, labels: np.ndarray, block_size: Tuple[int, int], padded_shape: Tuple[int, int, int]) -> np.ndarray:
    block_h, block_w = block_size
    padded_h, padded_w, channels = padded_shape
    num_blocks_per_row = padded_w // block_w
    image = np.zeros((padded_h, padded_w, channels), dtype=np.float32)

    for idx, label in enumerate(labels):
        y = (idx // num_blocks_per_row) * block_h
        x = (idx % num_blocks_per_row) * block_w
        block = codebook[label].reshape(block_h, block_w, channels)
        image[y:y + block_h, x:x + block_w, :] = block

    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def estimate_compression_ratio(original_shape: Tuple[int, int, int], codebook: np.ndarray, label_count: int, k: int) -> float:
    orig_bits = original_shape[0] * original_shape[1] * original_shape[2] * 8
    bits_per_index = max(1, math.ceil(math.log2(max(k, 2))))
    indices_bits = bits_per_index * label_count
    codebook_bits = codebook.astype(np.float32).nbytes * 8
    compressed_bits = indices_bits + codebook_bits
    return orig_bits / compressed_bits if compressed_bits else 0.0


def compress_flow():
    image_path = input("Enter path of image to compress: ").strip()
    image_array, mode = load_image(image_path)
    h, w, _ = image_array.shape
    print(f"Loaded image with shape {image_array.shape}")

    config_type = input("Specify block by (s)ize or by block (c)ount per dimension? [s/c]: ").strip().lower()
    if config_type == "c":
        blocks_h = read_positive_int("Number of blocks vertically: ")
        blocks_w = read_positive_int("Number of blocks horizontally: ")
        block_h = math.ceil(h / blocks_h)
        block_w = math.ceil(w / blocks_w)
    else:
        block_h = read_positive_int("Block height (pixels): ")
        block_w = read_positive_int("Block width (pixels): ")

    codebook_size = read_positive_int("Enter codebook size (e.g., 16, 32, 64): ")

    padded_image, pad = pad_image(image_array, (block_h, block_w))
    blocks = split_blocks(padded_image, (block_h, block_w))

    if codebook_size > len(blocks):
        print(f"Codebook size {codebook_size} exceeds number of blocks {len(blocks)}.")
        return

    codebook, labels = kmeans(blocks, codebook_size)
    padded_shape = padded_image.shape

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{Path(image_path).stem}_{timestamp}"

    codebook_path = outputs_dir / f"{base_name}_codebook.npy"
    np.save(codebook_path, codebook)

    compressed_path = outputs_dir / f"{base_name}_compressed.npz"
    np.savez_compressed(
        compressed_path,
        indices=labels,
        original_shape=np.array(image_array.shape, dtype=np.int32),
        padded_shape=np.array(padded_shape, dtype=np.int32),
        block_size=np.array([block_h, block_w], dtype=np.int32),
        padding=np.array(pad, dtype=np.int32),
        codebook_path=str(codebook_path.resolve()),
        image_mode=mode,
    )

    reconstructed = reconstruct_image(codebook, labels, (block_h, block_w), padded_shape)
    if pad[0]:
        reconstructed = reconstructed[:-pad[0], :, :]
    if pad[1]:
        reconstructed = reconstructed[:, :-pad[1], :]
    compressed_image_path = outputs_dir / f"{base_name}_compressed_preview.png"
    Image.fromarray(reconstructed).save(compressed_image_path)

    ratio = estimate_compression_ratio(image_array.shape, codebook, len(labels), codebook_size)

    print(f"Compression complete.")
    print(f"- Codebook saved to: {codebook_path}")
    print(f"- Compressed data saved to: {compressed_path}")
    print(f"- Preview image saved to: {compressed_image_path}")
    print(f"- Compression ratio ≈ {ratio:.2f}:1")


def decompress_flow():
    compressed_path = input("Enter path of compressed .npz file: ").strip()
    data = np.load(compressed_path, allow_pickle=True)
    indices = data["indices"]
    original_shape = tuple(data["original_shape"])
    padded_shape = tuple(data["padded_shape"])
    block_h, block_w = data["block_size"]
    pad_h, pad_w = data["padding"]
    codebook_path = Path(str(data["codebook_path"]))
    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook file not found at {codebook_path}")
    codebook = np.load(codebook_path)

    reconstructed = reconstruct_image(codebook, indices, (int(block_h), int(block_w)), tuple(padded_shape))
    if pad_h:
        reconstructed = reconstructed[:-pad_h, :, :]
    if pad_w:
        reconstructed = reconstructed[:, :-pad_w, :]

    reconstructed = reconstructed[: original_shape[0], : original_shape[1], :]
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = outputs_dir / f"decompressed_{timestamp}.png"
    Image.fromarray(reconstructed).save(output_path)

    print(f"Decompression complete.")
    print(f"- Decompressed image saved to: {output_path}")


def main():
    while True:
        choice = prompt_menu()
        if choice == "1":
            try:
                compress_flow()
            except Exception as exc:
                print(f"Compression failed: {exc}")
        elif choice == "2":
            try:
                decompress_flow()
            except Exception as exc:
                print(f"Decompression failed: {exc}")
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

