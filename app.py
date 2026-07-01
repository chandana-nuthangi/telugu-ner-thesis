"""
Telugu NER Streamlit App
Fine-tuned DistilBERT-multilingual on Naamapadam Telugu (F1 0.78 on test set).
Part of the from-floor-to-data portfolio.
"""

import io
import json
import time
from pathlib import Path

import pandas as pd
import pdfplumber
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telugu NER — Transfer Learning for Low-Resource NLP",
    page_icon="🔤",
    layout="wide",
)

MODEL_PATH = "./models/distilbert-telugu-ner"

ENTITY_COLORS = {
    "PER": "#ffd6a5",   # peach — person
    "ORG": "#a5d8ff",   # blue — organization
    "LOC": "#b9fbc0",   # green — location
}

ENTITY_LABELS = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
}


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Telugu NER model (first time takes ~15 sec)...")
def load_model():
    from transformers import pipeline
    return pipeline(
        "token-classification",
        model=MODEL_PATH,
        aggregation_strategy="first",  # better for morphologically rich languages
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_pdf_text(file_bytes: bytes) -> list[dict]:
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Split long text at sentence-ish boundaries to respect 128-token limit."""
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current.strip():
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Run NER
# ---------------------------------------------------------------------------
def run_ner_on_text(text: str, page: int = 1) -> list[dict]:
    """Return a list of entity dicts from text."""
    nlp = load_model()
    rows = []
    for chunk in chunk_text(text):
        for ent in nlp(chunk):
            word = ent["word"].replace("##", "").strip()
            if len(word) > 1:  # filter noise
                rows.append({
                    "entity": word,
                    "type": ent["entity_group"],
                    "confidence": round(float(ent["score"]), 3),
                    "page": page,
                })
    return rows


def highlight_entities(text: str, entities: pd.DataFrame) -> str:
    """Wrap entity mentions in colored spans."""
    html = text
    # longest first so substrings don't break longer matches
    unique = entities.drop_duplicates("entity").sort_values(
        "entity", key=lambda s: s.str.len(), ascending=False
    )
    for _, row in unique.head(100).iterrows():
        color = ENTITY_COLORS.get(row["type"], "#dddddd")
        ent = row["entity"]
        if not ent or ent not in html:
            continue
        html = html.replace(
            ent,
            f'<mark style="background:{color};padding:2px 5px;'
            f'border-radius:4px;margin:0 2px">{ent}'
            f'<sub style="font-size:0.7em;margin-left:3px;color:#555">'
            f'{row["type"]}</sub></mark>',
        )
    return html.replace("\n", "<br>")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔤 Telugu Named Entity Recognition")
st.caption(
    "Fine-tuned DistilBERT-multilingual on the Naamapadam Telugu benchmark "
    "(Mhaske et al., ACL 2023). "
    "**F1 = 0.78** on the manually annotated test set, "
    "with a 66M-parameter model — 40% smaller than BERT."
)

# Model verification
if not Path(MODEL_PATH).exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Please make sure the trained model files are in the models/ folder."
    )
    st.stop()

tab_input, tab_upload, tab_about = st.tabs([
    "✍️ Type / paste text",
    "📄 Upload PDF",
    "ℹ️ About the model",
])

# ------------------------------ Tab 1: Text ------------------------------
with tab_input:
    default_text = (
        "నరేంద్ర మోదీ న్యూ ఢిల్లీలో భారత ప్రభుత్వ కార్యాలయంలో మాట్లాడారు. "
        "హైదరాబాద్ నుండి రవీంద్ర రెడ్డి కూడా ఈ సమావేశంలో పాల్గొన్నారు."
    )
    text_input = st.text_area(
        "Enter Telugu text",
        value=default_text,
        height=140,
    )

    if st.button("Extract entities", type="primary", key="btn_text"):
        if not text_input.strip():
            st.warning("Please enter some Telugu text.")
        else:
            t0 = time.perf_counter()
            entities = run_ner_on_text(text_input)
            elapsed = time.perf_counter() - t0

            df = pd.DataFrame(entities, columns=["entity", "type", "confidence", "page"])

            if df.empty:
                st.info("No entities detected in this text.")
            else:
                # metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entities found", len(df))
                m2.metric("Unique", df["entity"].nunique())
                m3.metric("Types", df["type"].nunique())
                m4.metric("Time", f"{elapsed*1000:.0f} ms")

                # highlighted preview
                st.markdown("### Highlighted text")
                st.markdown(
                    f'<div style="border:1px solid #ccc;border-radius:8px;'
                    f'padding:16px;line-height:2;font-size:1.05em">'
                    f'{highlight_entities(text_input, df)}</div>',
                    unsafe_allow_html=True,
                )

                # legend
                legend = " ".join(
                    f'<span style="background:{c};padding:2px 8px;border-radius:4px">'
                    f'{ENTITY_LABELS[t]} ({t})</span>'
                    for t, c in ENTITY_COLORS.items()
                )
                st.markdown(f"**Legend:** {legend}", unsafe_allow_html=True)

                # table + downloads
                st.markdown("### Extracted entities")
                st.dataframe(df.drop(columns=["page"]), use_container_width=True)

                c1, c2 = st.columns(2)
                c1.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="telugu_entities.csv",
                    mime="text/csv",
                )
                c2.download_button(
                    "⬇️ Download JSON",
                    json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name="telugu_entities.json",
                    mime="application/json",
                )

# ------------------------------ Tab 2: PDF ------------------------------
with tab_upload:
    uploaded = st.file_uploader("Upload a PDF containing Telugu text", type=["pdf"])

    if uploaded:
        pages = extract_pdf_text(uploaded.getvalue())
        if not pages:
            st.error(
                "No extractable text found. This PDF might be scanned. "
                "Try adding an OCR step first."
            )
        else:
            st.success(f"Extracted {len(pages)} page(s).")

            t0 = time.perf_counter()
            all_entities = []
            progress = st.progress(0.0, "Running NER on each page...")
            for i, page in enumerate(pages):
                all_entities.extend(run_ner_on_text(page["text"], page=page["page"]))
                progress.progress((i + 1) / len(pages))
            progress.empty()
            elapsed = time.perf_counter() - t0

            df = pd.DataFrame(all_entities, columns=["entity", "type", "confidence", "page"])

            if df.empty:
                st.info("No entities detected in this document.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entities", len(df))
                m2.metric("Unique", df["entity"].nunique())
                m3.metric("Types", df["type"].nunique())
                m4.metric("Time", f"{elapsed:.1f} s")

                st.markdown("### Type distribution")
                st.bar_chart(df["type"].value_counts())

                st.markdown("### All extracted entities")
                st.dataframe(df, use_container_width=True, height=350)

                c1, c2 = st.columns(2)
                c1.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="telugu_entities.csv",
                    mime="text/csv",
                )
                c2.download_button(
                    "⬇️ Download JSON",
                    json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False).encode("utf-8"),
                    file_name="telugu_entities.json",
                    mime="application/json",
                )

# ------------------------------ Tab 3: About ------------------------------
with tab_about:
    st.markdown(
        """
### The problem

Telugu has ~80 million speakers but remains **low-resource for NLP** — labelled
datasets, pretraining corpora, and benchmarks are sparse. Existing SOTA Telugu
NER models (IndicNER, MuRIL fine-tuned) achieve ~0.80 F1 but require large
(180–240M parameter) models that are expensive to deploy in production.

### The question

Can a **smaller, faster, distilled multilingual model** deliver acceptable
Telugu NER performance via transfer learning?

### The approach

- **Model**: `distilbert-base-multilingual-cased` (66M parameters, 40% smaller than BERT)
- **Dataset**: [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (Mhaske et al., ACL 2023) — largest publicly available Indic NER corpus
- **Method**: Fine-tuning for token classification with proper subword-to-word label alignment
- **Metric**: Entity-level F1 via seqeval on the manually annotated test set

### The result

| Metric | Value |
|---|---|
| **F1 (test set)** | **0.78** |
| Precision | 0.78 |
| Recall | 0.78 |
| Model size | 66M params (2.7× smaller than IndicNER) |
| Training data | 50k sentences (10% of Naamapadam Telugu) |
| Training time | 10 min on Colab T4 |

**Within 2 F1 points of published SOTA** with a much smaller model on 10% of the data.

### References
- Sanh et al. (2019). *DistilBERT.* arXiv:1910.01108
- Khanuja et al. (2021). *MuRIL.* arXiv:2103.10730
- Mhaske et al. (2023). *Naamapadam.* ACL 2023
        """
    )