"""Index PDF manuals for retrieval.

Converts PDFs → page images → Qwen3-VL-Embedding-2B embeddings → numpy index.

Usage:
  python scripts/index_documents.py --pdf_dir data/manuals/ --output_dir data/index/
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF pages to PIL Images."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def encode_with_teacher(
    images: list[Image.Image],
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    batch_size: int = 4,
) -> np.ndarray:
    """Encode document page images using the Qwen3-VL-Embedding-2B teacher.

    Uses the official `sentence-transformers` entry point documented on the
    model card — it wraps the custom Qwen3-VL embedder correctly. Requires
    sentence-transformers>=5.2 (for `sentence_transformers.base`).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading teacher: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  dim={model.get_embedding_dimension()}, device={model.device}")

    # ST expects image inputs as dicts `{"image": PIL.Image}`.
    inputs = [{"image": img} for img in images]

    embeddings = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory with PDF manuals")
    parser.add_argument("--output_dir", type=str, default="data/index")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="SentenceTransformer-compatible teacher model name.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    pdf_files = sorted([f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDF files")

    all_images: list[Image.Image] = []
    page_metadata: list[dict] = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.pdf_dir, pdf_file)
        print(f"Processing {pdf_file}...")
        pages = pdf_to_images(pdf_path, dpi=args.dpi)

        for page_num, img in enumerate(pages):
            page_id = f"{pdf_file}_p{page_num:04d}"
            img_filename = f"{page_id}.png"
            img.save(os.path.join(args.image_dir, img_filename))

            all_images.append(img)
            page_metadata.append({
                "page_id": page_id,
                "image_path": img_filename,
                "source": pdf_file,
                "page_number": page_num,
            })

    print(f"\nTotal: {len(all_images)} pages from {len(pdf_files)} PDFs")

    print(f"\nEncoding with {args.teacher_model}...")
    embeddings = encode_with_teacher(
        all_images,
        model_name=args.teacher_model,
        batch_size=args.batch_size,
    )

    np.save(os.path.join(args.output_dir, "doc_embeddings.npy"), embeddings.astype(np.float16))
    with open(os.path.join(args.output_dir, "page_metadata.json"), "w") as f:
        json.dump(page_metadata, f, indent=2)

    print(f"\nIndex saved to {args.output_dir}/")
    print(f"  doc_embeddings.npy: {embeddings.shape}")
    print(f"  page_metadata.json: {len(page_metadata)} pages")
    print(f"  Images saved to {args.image_dir}/")


if __name__ == "__main__":
    main()
