# Telugu NER — Transfer Learning for a Low-Resource Language

Fine-tuning DistilBERT-multilingual for Named Entity Recognition on Telugu, with a deployed Streamlit app for PDF entity extraction.

**Result: F1 = 0.78** on Naamapadam test set — within 2 points of published SOTA (IndicNER), using a **2.7× smaller model** trained on **10% of the data**.

---

## The research problem

Telugu has ~80 million speakers but remains **low-resource for NLP** — labelled datasets, pretraining corpora, and benchmarks are sparse. Existing SOTA Telugu NER models (IndicNER, MuRIL fine-tuned) achieve ~0.80 F1 but require 180–240M parameter models that are expensive to deploy at scale. This raises a research question about **model efficiency tradeoffs for low-resource NLP**: how much accuracy is lost when the model is heavily compressed via distillation, and is that loss acceptable for practical deployment?

**The question this research answers:** Can a smaller, faster, distilled multilingual model deliver acceptable Telugu NER performance through transfer learning?

## Approach

- **Model:** `distilbert-base-multilingual-cased` (66M parameters, 40% smaller than BERT)
- **Dataset:** [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (Mhaske et al., ACL 2023) — largest publicly available Indic NER corpus, 507k Telugu sentences
- **Method:** Fine-tuning for token classification with subword-to-word label alignment (3 epochs, learning rate 2e-5, batch size 32, fp16)
- **Metric:** Entity-level F1 via seqeval on the 847-sentence manually annotated test set
- **Baseline landscape:** IndicNER (ACL 2023 SOTA, IndicBERT-based), MuRIL, mBERT, XLM-R

## Results

| Metric | Value |
|---|---|
| **F1 (test set)** | **0.7803** |
| Precision | 0.7827 |
| Recall | 0.7779 |
| Accuracy | 0.9389 |
| Model size | 66M params |
| Training data used | 50k / 507k sentences (10%) |
| Training time | 10 min 42 sec on Colab T4 |
| Inference latency | ~50ms per sentence (CPU) |

Trained on only 10% of Naamapadam Telugu, the fine-tuned DistilBERT lands within 2 F1 points of IndicNER while being 2.7× smaller.

## Demo

The Streamlit app supports:
- ✍️ **Text input** — paste Telugu text, get highlighted entities
- 📄 **PDF upload** — extract entities from Telugu PDFs
- ⬇️ **Downloadable results** as CSV or JSON

### Example: Political / General Text

*Input:* "నరేంద్ర మోదీ న్యూ ఢిల్లీలో భారత ప్రభుత్వ కార్యాలయంలో మాట్లాడారు"

*Output:* Correctly identifies "నరేంద్ర మోదీ" (Narendra Modi) → PER (0.90), "న్యూ ఢిల్లీలో" (New Delhi) → LOC (0.73)

## Known Limitations (Honest Error Analysis)

**Domain sensitivity.** Naamapadam training data is auto-projected from English news, which skews toward political and geographic entities. Testing on a Telugu space journalism article (`demo/Telugu_document.pdf`), the model:
- ✅ Correctly identifies general geographic entities (India, Moon)
- ⚠️ Misses domain-specific entities like "ISRO", "Skyroot Aerospace", "Vikram-1" (space industry vocabulary underrepresented in training)

**Production implications.** For domain-specific Telugu NER (finance, tech, healthcare), this model would need either (a) fine-tuning on domain-annotated data, (b) a gazetteer of known entities, or (c) hybrid approach with LLM few-shot prompting for rare domains.

**Training data volume.** Only 10% of Naamapadam Telugu was used to keep training under 15 minutes. Full-data run (~90 min) would likely push F1 to ~0.80.

## Tech Stack

Python · PyTorch · HuggingFace `transformers` / `datasets` · `seqeval` · `pdfplumber` · Streamlit · pandas

## Repository Structure

```
telugu-ner-thesis/
├── app.py                     # Streamlit application
├── notebooks/
│   └── train_finetune.ipynb   # Colab training notebook (fine-tuning)
├── demo/
│   └── Telugu_document.pdf    # Sample PDF for testing
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/chandana-nuthangi/telugu-ner-thesis.git
cd telugu-ner-thesis
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Download the fine-tuned model (hosted separately due to size)
# Instructions: see MODEL.md

streamlit run app.py
```

## Anchor References

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108
2. Khanuja, S., et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730
3. Mhaske, A., et al. (2023). *Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages.* ACL 2023, arXiv:2212.10168

## About

Built by [Chandana Nuthangi](https://github.com/chandana-nuthangi). M.Sc. Data Science thesis.

Fine-tuned model available on [HuggingFace Hub](https://huggingface.co/chandanaau/distilbert-telugu-ner).
