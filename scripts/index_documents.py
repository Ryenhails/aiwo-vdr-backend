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


def encode_with_teacher(images: list[Image.Image], batch_size: int = 4):
    """Encode document page images with Qwen3-VL-Embedding-2B."""
    import torch
    from transformers import AutoProcessor

    # Use the official Qwen3-VL-Embedding script
    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
    ).eval().cuda()

    all_embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="Encoding pages"):
        batch_imgs = images[i:i + batch_size]
        # Format as conversations for Qwen3-VL
        conversations = []
        for img in batch_imgs:
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "image", "image": img}]},
            ])

        text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        from qwen_vl_utils.vision_process import process_vision_info
        imgs_processed, _, video_kwargs = process_vision_info(
            conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True,
        )

        inputs = processor(
            text=text, images=imgs_processed, padding=True, truncation=True,
            max_length=8192, return_tensors="pt", do_resize=False, **video_kwargs,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            # Last-token pooling
            attn = inputs["attention_mask"]
            flipped = attn.flip(dims=[1])
            last_pos = flipped.argmax(dim=1)
            col = attn.shape[1] - last_pos - 1
            row = torch.arange(outputs.last_hidden_state.shape[0], device="cuda")
            embs = outputs.last_hidden_state[row, col]
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
            all_embeddings.append(embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory with PDF manuals")
    parser.add_argument("--output_dir", type=str, default="data/index")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)

    # Collect all PDFs
    pdf_files = sorted([f for f in os.listdir(args.pdf_dir) if f.lower().endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDF files")

    all_images = []
    page_metadata = []

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

    # Encode with teacher
    print("\nEncoding with Qwen3-VL-Embedding-2B...")
    embeddings = encode_with_teacher(all_images, batch_size=args.batch_size)

    # Save index
    np.save(os.path.join(args.output_dir, "doc_embeddings.npy"), embeddings.astype(np.float16))
    with open(os.path.join(args.output_dir, "page_metadata.json"), "w") as f:
        json.dump(page_metadata, f, indent=2)

    print(f"\nIndex saved to {args.output_dir}/")
    print(f"  doc_embeddings.npy: {embeddings.shape}")
    print(f"  page_metadata.json: {len(page_metadata)} pages")
    print(f"  Images saved to {args.image_dir}/")


if __name__ == "__main__":
    main()
