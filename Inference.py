import os
import io
import json

import fitz
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests


class ContextualPDFClassifier:
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL = "openai/gpt-4o"

    CATEGORIES = [
        "Sensitive PII Data",
        "Confidential Company Info",
        "Safe for Public Use",
        "Unsafe Language and Content"
    ]

    PAGE_SYSTEM_PROMPT = """
You are classifying ALL entities on a single PDF page.

Each page contains:
- text blocks (with id, bbox, text)
- images (with id, bbox, caption)

You MUST examine BOTH text blocks AND images.
Do NOT ignore any image. Do NOT skip image entities.

You MUST classify every entity into one or more of ONLY these categories:
- Sensitive PII Data
- Confidential Company Info
- Safe for Public Use
- Unsafe Language and Content

CONTEXTUAL CLASSIFICATION RULES:
- Treat images as likely company property. Do NOT classify as unsafe based solely on generic image content.
- Images should inherit context from surrounding text. For example, technical details in text may render associated images confidential.
- If nearby text indicates confidentiality, the associated image inherits CONFIDENTIALITY.
- An image with harmless content may still be UNSAFE or CONFIDENTIAL depending on text.
- You MUST classify ALL images, even if the caption is short or general.
- For each image, decide whether it belongs to any NON-SAFE category.

OUTPUT REQUIREMENTS (STRICT JSON ONLY):

{
  "page_index": int,
  "issues": [
    {
      "entity_id": "<id of text block or image>",
      "entity_type": "text" or "image",
      "bbox": [x0, y0, x1, y1],
      "caption": "<caption text for images, empty string for text>",
      "categories": ["..."],
      "rationale": "2–4 sentences explaining classification.",
      "quotes": ["verbatim substrings for text blocks, empty list for images"]
    }
  ]
}

RULES:
- Include ONLY entities whose categories DO NOT include "Safe for Public Use".
- For text, `quotes` MUST contain exact verbatim substrings.
- For images, `quotes` MUST be an empty list.
- `caption` must be EXACTLY the provided caption for images; empty for text.
- Do NOT omit image entities. If an image is safe, omit it entirely from `issues`. If NOT safe, include it.
- STRICT JSON ONLY. NO commentary, no markdown.
"""

    def __init__(self, openrouter_api_key: str | None = None, model: str | None = None):
        # API key
        self.OPENROUTER_API_KEY = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY in .env file")

        # Model override (optional)
        if model:
            self.MODEL = model

        # Load BLIP captioning model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    # ============================================================
    # OpenRouter API Wrapper
    # ============================================================
    def call_openrouter(self, messages, max_tokens=1500):
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Contextual PDF Classifier"
        }

        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens
        }

        resp = requests.post(self.OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ============================================================
    # STEP 1 — Extract text blocks
    # ============================================================
    def extract_text_blocks(self, pdf_path):
        doc = fitz.open(pdf_path)
        pages = []

        for pno in range(len(doc)):
            page = doc[pno]
            raw_blocks = page.get_text("blocks")
            blocks = []

            for idx, blk in enumerate(raw_blocks):
                if len(blk) < 5:
                    continue
                x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
                norm = " ".join(text.split())
                if not norm:
                    continue

                blocks.append({
                    "id": f"t{pno}-{idx}",
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": norm
                })

            pages.append({
                "page_index": pno,
                "blocks": blocks
            })

        doc.close()
        return pages

    # ============================================================
    # STEP 2 — Extract image captions + bounding boxes
    # ============================================================
    def extract_image_data(self, pdf_path):
        pdf = fitz.open(pdf_path)
        results = []

        for pno, page in enumerate(pdf):
            images = page.get_images(full=True)

            for idx, img in enumerate(images):
                xref = img[0]
                base = pdf.extract_image(xref)
                im_bytes = base["image"]

                pil = Image.open(io.BytesIO(im_bytes)).convert("RGB")

                # ignore small logos
                if pil.width < 100 or pil.height < 100:
                    continue

                # BLIP caption
                inp = self.processor(pil, return_tensors="pt")
                out = self.blip_model.generate(**inp)
                caption = self.processor.decode(out[0], skip_special_tokens=True)

                bbox = page.get_image_bbox(img)

                results.append({
                    "id": f"img{pno}-{idx}",
                    "page_index": pno,
                    "caption": caption,
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                })

        pdf.close()
        return results

    # ============================================================
    # STEP 3 — Contextual page classifier
    # ============================================================
    def classify_page_contextually(self, page_text, images):
        pno = page_text["page_index"]

        # only images on this page
        imgs = [
            {"id": img["id"], "bbox": img["bbox"], "caption": img["caption"]}
            for img in images if img["page_index"] == pno
        ]

        payload = {
            "page_index": pno,
            "text_blocks": page_text["blocks"],
            "images": imgs
        }

        raw = self.call_openrouter([
            {"role": "system", "content": self.PAGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]).strip()

        # isolate json
        first, last = raw.find("{"), raw.rfind("}")
        if first != -1 and last != -1:
            raw = raw[first:last+1]

        try:
            parsed = json.loads(raw)
        except:
            parsed = {"page_index": pno, "issues": []}

        # filter safe categories
        clean = []
        for issue in parsed.get("issues", []):
            issue["categories"] = [
                c for c in issue.get("categories", [])
                if c in self.CATEGORIES and c != "Safe for Public Use"
            ]
            if issue["categories"]:
                clean.append(issue)

        parsed["issues"] = clean
        return parsed

    # ============================================================
    # STEP 4 — Extract image-only results
    # ============================================================
    def extract_image_results(self, page_results):
        out = []
        for page in page_results:
            for issue in page["issues"]:
                if issue["entity_type"] == "image":
                    out.append(issue)
        return out

    # ============================================================
    # STEP 5 — Highlight text + image regions
    # ============================================================
    def highlight_pdf(self, pdf_path, page_results, output_pdf):
        doc = fitz.open(pdf_path)

        for page in page_results:
            pno = page["page_index"]
            pg = doc[pno]

            for issue in page["issues"]:
                bbox = issue["bbox"]
                rect = fitz.Rect(*bbox)

                if issue["entity_type"] == "text":
                    for q in issue.get("quotes", []):
                        hits = pg.search_for(q)
                        for h in hits:
                            # Check if hit bounding box is valid before highlighting
                            if h.is_valid and not h.is_empty and not h.is_infinite:
                                annot = pg.add_highlight_annot(h)
                                annot.set_info(title="PDF-Classifier",
                                content=f"Categories: {', '.join(issue.get('categories'))}\n{issue.get('rationale')[:400]}")
                                annot.update()
                            else:
                                print(f"Skipping invalid text highlight bbox on page {pno}: {h}")


                elif issue["entity_type"] == "image":
                    # Check if image bounding box is valid before annotating
                    if rect.is_valid and not rect.is_empty and not rect.is_infinite:
                        annot = pg.add_rect_annot(rect)
                        annot.set_colors({"stroke": (1, 0, 0)})
                        annot.set_info(title="PDF-Classifier",
                                content=f"Categories: {', '.join(issue.get('categories'))}\n{issue.get('rationale')[:400]}")
                        annot.set_border(width=2)
                        annot.update()
                    else:
                        print(f"Skipping invalid image bbox on page {pno}: {bbox}")


        doc.save(output_pdf)
        doc.close()

    # ============================================================
    # STEP 6 — Final classification
    # ============================================================
    def aggregate_final(self, page_results):
        cats = set()
        for page in page_results:
            for issue in page["issues"]:
                cats.update(issue["categories"])

        if not cats:
            return {"final_categories": ["Safe for Public Use"]}

        return {"final_categories": sorted(cats)}

    # ============================================================
    # ✅ MASTER FUNCTION — CALL THIS FROM BACKEND
    # ============================================================
    def classify_pdf_contextually(
        self,
        pdf_path: str,
        output_json="contextual_classification.json",
        output_pdf="contextual_highlights.pdf"
    ):
        # Extract components
        text_pages = self.extract_text_blocks(pdf_path)
        images = self.extract_image_data(pdf_path)

        # Page-by-page classification
        page_results = []
        for p in text_pages:
            page_results.append(self.classify_page_contextually(p, images))

        # extract image-level issues
        image_results = self.extract_image_results(page_results)

        # final result
        final = self.aggregate_final(page_results)

        final_json = {
            "page_results": page_results,
            "image_results": image_results,
            "final_classification": final
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        # highlight offending regions
        self.highlight_pdf(pdf_path, page_results, output_pdf)

        return {
            **final_json,
            "highlighted_pdf_path": output_pdf,
            "json_path": output_json
        }
