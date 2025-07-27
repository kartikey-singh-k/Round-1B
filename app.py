import os
import json
import fitz  # PyMuPDF
import time
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("models/MiniLM")

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            for para in text.split("\n\n"):
                cleaned = para.strip().replace('\n', ' ')
                if len(cleaned) > 50:
                    chunks.append({
                        "page": page_num,
                        "text": cleaned
                    })
    return chunks

def process_collection(collection_path):
    input_json_path = os.path.join(collection_path, "challenge1b_input.json")
    output_json_path = os.path.join(collection_path, "challenge1b_output.json")
    pdf_dir = os.path.join(collection_path, "PDFs")

    with open(input_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    persona = config["persona"]
    job = config["job"]
    input_docs = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

    query = f"{persona}. Task: {job}"
    query_emb = model.encode(query, convert_to_tensor=True)

    results = []
    for pdf_file in input_docs:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        chunks = extract_chunks_from_pdf(pdf_path)

        for chunk in chunks:
            chunk_text = chunk["text"]
            score = util.cos_sim(model.encode(chunk_text, convert_to_tensor=True), query_emb)[0][0].item()

            results.append({
                "document": pdf_file,
                "page_number": chunk["page"],
                "section_title": chunk_text[:80] + ("..." if len(chunk_text) > 80 else ""),
                "refined_text": chunk_text,
                "importance_rank": score
            })

    # Sort by importance descending and pick top N
    results = sorted(results, key=lambda x: x["importance_rank"], reverse=True)[:15]

    output = {
        "metadata": {
            "input_documents": input_docs,
            "persona": persona,
            "job": job,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "extracted_sections": [
            {
                "document": r["document"],
                "page_number": r["page_number"],
                "section_title": r["section_title"],
                "importance_rank": round(r["importance_rank"], 4)
            } for r in results
        ],
        "subsection_analysis": [
            {
                "document": r["document"],
                "page_number": r["page_number"],
                "refined_text": r["refined_text"]
            } for r in results
        ]
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    for folder in os.listdir(base_path):
        collection_path = os.path.join(base_path, folder)
        if os.path.isdir(collection_path) and "challenge1b_input.json" in os.listdir(collection_path):
            print(f"📂 Processing: {folder}")
            process_collection(collection_path)
