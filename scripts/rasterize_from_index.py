"""Rebuild page PNGs from the committed index — CPU-only, no GPU required.

Reads `data/index/page_metadata.json`, looks up each page's source PDF in
`data/manuals/`, and writes a PNG per page into `data/images/` at the given
DPI. Used after a fresh clone because `data/images/` is gitignored
(~188 MB for the Ponsse manual).

Usage:
    python scripts/rasterize_from_index.py
    python scripts/rasterize_from_index.py --dpi 150
"""

import argparse
import json
import os
import sys

import fitz  # PyMuPDF


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index_dir",   default="data/index")
    ap.add_argument("--manual_dir",  default="data/manuals")
    ap.add_argument("--image_dir",   default="data/images")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Must match the DPI used during index_documents.py "
                         "if you want page images aligned with the embeddings.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-rasterise pages even if the PNG already exists.")
    args = ap.parse_args()

    meta_path = os.path.join(args.index_dir, "page_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found — did you commit the index?",
              file=sys.stderr)
        return 1

    with open(meta_path) as f:
        metadata: list[dict] = json.load(f)

    os.makedirs(args.image_dir, exist_ok=True)

    by_pdf: dict[str, list[dict]] = {}
    for m in metadata:
        by_pdf.setdefault(m["source"], []).append(m)

    scale = args.dpi / 72.0
    mat = fitz.Matrix(scale, scale)

    total = len(metadata)
    written = 0
    skipped = 0

    for pdf_name, pages in by_pdf.items():
        pdf_path = os.path.join(args.manual_dir, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"ERROR: {pdf_path} missing — cannot rasterise its pages",
                  file=sys.stderr)
            return 2

        print(f"Processing {pdf_name} ({len(pages)} pages) ...")
        doc = fitz.open(pdf_path)
        try:
            for m in pages:
                out_path = os.path.join(args.image_dir, m["image_path"])
                if os.path.exists(out_path) and not args.overwrite:
                    skipped += 1
                    continue
                pix = doc[m["page_number"]].get_pixmap(matrix=mat)
                pix.save(out_path)
                written += 1
        finally:
            doc.close()

    print(f"\nDone. total={total} written={written} skipped={skipped} "
          f"dpi={args.dpi} → {args.image_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
