"""Relabel the legend of Figure 2 for the IJF R2 revision (Reviewer 2, comment 1).

The panels are unchanged. Only the legend at the bottom right of
``figures/fig2_updated.pdf`` is replaced:
    observed / prediction / extrapolation
    -> observed / within training horizon / beyond training horizon
Patch colours are the rendered colours of the original legend patches.

Requires PyMuPDF (``pip install pymupdf``).
Usage: python examples/r2_fig2_relabel.py AFM_R2_revision/figures/fig2_updated.pdf AFM_R2_revision/figures/fig2_r2.pdf
"""
import sys

import pymupdf

LABELS = [
    ("observed", (245, 245, 245)),
    ("within training horizon", (213, 239, 249)),
    ("beyond training horizon", (254, 251, 211)),
]
OLD = ("observed", "prediction", "extrapolation")
FONT, SIZE = "helv", 14.0
PATCH_W, PATCH_H, GAP_PATCH, GAP_ITEM = 43.0, 12.0, 4.0, 14.0


def main(src: str, dst: str) -> None:
    doc = pymupdf.open(src)
    page = doc[0]
    spans = {
        s["text"].strip(): pymupdf.Rect(s["bbox"])
        for b in page.get_text("dict")["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip() in OLD
    }
    assert set(spans) == set(OLD), f"legend text not found: {sorted(spans)}"
    patch_y0 = 459.5  # original patch rectangles span y 459.5-471.5
    right = spans["extrapolation"].x1
    band = pymupdf.Rect(page.rect.x0 + 800, spans["prediction"].y0 - 2, page.rect.x1, page.rect.y1)

    # Remove the old legend (text and patches) inside the band.
    page.add_redact_annot(band, fill=(1, 1, 1))
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)

    widths = [pymupdf.get_text_length(t, fontname=FONT, fontsize=SIZE) for t, _ in LABELS]
    total = sum(PATCH_W + GAP_PATCH + w for w in widths) + GAP_ITEM * (len(LABELS) - 1)
    x = right - total
    baseline = patch_y0 + PATCH_H - 1.0
    for (text, rgb), w in zip(LABELS, widths):
        page.draw_rect(pymupdf.Rect(x, patch_y0, x + PATCH_W, patch_y0 + PATCH_H),
                       color=None, fill=[c / 255 for c in rgb], width=0)
        x += PATCH_W + GAP_PATCH
        page.insert_text((x, baseline), text, fontname=FONT, fontsize=SIZE, color=(0, 0, 0))
        x += w + GAP_ITEM
    doc.save(dst, garbage=3, deflate=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
