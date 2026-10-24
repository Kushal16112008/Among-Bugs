# CampusRules AI — College Regulations Assistant

A fast hackathon MVP using Google's hosted Gemma API + RAG over official college PDF regulations. **This version does not use Ollama.**

## Features
- Upload one or more PDFs
- Extract page-aware text with PyMuPDF
- Split pages into overlapping passages
- Retrieve passages using Sentence Transformers embeddings and cosine similarity
- Send retrieved evidence to hosted Gemma through Google's Gen AI SDK
- Show source PDF, page number, excerpt, and similarity score
- Decline to answer when retrieval evidence is weak

## Requirements
- Windows 10/11
- Python 3.10+ recommended
- Internet access for package, embedding model, and hosted API requests
- Google AI Studio API key with access to the selected hosted Gemma model

## Run on Windows

### 1. Get an API key
Create a Google AI Studio API key:
https://aistudio.google.com/apikey

Keep the key private. Do not commit it to GitHub or paste it into source code. The app asks for it in a password field in the sidebar.

### 2. Create and activate a virtual environment

From this folder in VS Code:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the app

```powershell
streamlit run app.py
```

A browser tab should open. Paste your API key in the sidebar, upload the college regulations PDF, and ask a question.

## Model
Default hosted Gemma model ID: `gemma-4-26b-a4b-it`. Google's hosted Gemma API currently documents Gemma 4 models. If this model is not enabled for your account or is unavailable, check the current official Gemma API model list and update the model ID in the sidebar.

## First run notes
- Sentence Transformers downloads `all-MiniLM-L6-v2` on first use. This is separate from the hosted Gemma API.
- Scanned/image-only PDFs may yield no text. Use a text-based PDF or OCR it before uploading.
- Internet is needed for hosted inference and first-time model downloads.
- This MVP stores its index in Streamlit session memory; it rebuilds when the uploaded PDFs change.
- Similarity threshold is a simple demo safeguard, not a calibrated confidence probability.

## Demo questions
Ask questions whose answers you can locate in the uploaded document, for example:
- "What does the document say about attendance?"
- "What are the semester exam eligibility requirements?"
- "What is the revaluation procedure?"
- "Does this document specify the deadline for submitting an application?"

Do not use invented example rules. Verify every demo answer against the actual PDF.

## Architecture
PDF upload -> PyMuPDF page extraction -> overlapping text chunks -> Sentence Transformers embeddings -> cosine-similarity retrieval -> hosted Gemma API -> answer + evidence citations.

## Important limitations
This is a prototype, not an official source of institutional decisions. Rules may differ by academic year/program. The assistant should clearly state when the uploaded documents do not establish an answer. Confirm important decisions with the college office.
