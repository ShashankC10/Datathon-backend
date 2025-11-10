# 🧠 Hitachi AI-Powered Regulatory Document Classifier — Multi-Modal AI Document Classification System

## 📄 Overview
**Compliance Checker** is an AI-powered multi-modal document auditing system that automatically classifies business documents (PDFs, images, memos, etc.) into compliance categories.  
It combines **text parsing**, **image captioning**, and **large language model (LLM)** reasoning to detect potential violations and generate explainable, citation-based outputs.

This project was developed as part of a Datathon challenge to automate **document compliance review** for organizations handling sensitive internal and external documents.

---

## 🚀 Key Features

### 🔍 Multi-Modal Input
- Accepts **PDF files** containing both text and embedded images.
- Extracts and processes:
  - **Text content** using PDF parsers.
  - **Images** using **BLIP image captioning** to generate descriptive text.
- Merges text and image context for holistic compliance analysis.

### ⚙️ Dynamic Prompt Library
- The system uses a **configurable Prompt Library** that defines the rules, categories, and contextual checks used during inference.
- Prompts are stored in a **dynamic structure**, allowing:
  - Easy modification of existing rules.
  - Creation of new compliance categories via the user interface.
  - Fine-tuning of language model reasoning using domain-specific knowledge.
- Example:
  ```json
  {
    "category": "Confidential",
    "criteria": [
      "Contains personal or employee data",
      "Includes trade secrets or proprietary information"
    ]
  }
  ```
- These prompts form **Prompt Trees**, which dynamically adapt based on user feedback and prior classification patterns.

### 🧠 Human-in-the-Loop (HITL)
- The app integrates **Human-in-the-Loop learning**, ensuring that SMEs (Subject Matter Experts) remain central to improving AI performance.
- Users can:
  - Validate or override classification outcomes.
  - Add new contextual rules directly through the **interactive web interface**.
  - Append new categories or redefine compliance standards.
- All feedback is stored and incorporated into the next inference cycle, making the system **continuously adaptive** and **organization-aware**.

### 💬 Interactive Web Interface
- Built using **Flask templates** for a smooth and intuitive user experience.
- Allows users to:
  - Upload documents.
  - Review classification summaries.
  - Add or edit compliance rules dynamically.
  - Provide structured feedback for retraining and fine-tuning.

### 📊 AI-Driven Inference Pipeline
- Uses **OpenRouter / Cloud LLM APIs** for semantic and contextual understanding.
- Incorporates **BLIP image captioning** for visual context enrichment.
- Generates **citation-based results**, linking each decision to specific document sections or visual content.
- Supports multiple compliance categories such as:
  - Public
  - Internal
  - Confidential
  - Restricted
  - Non-Compliant

---

## 🧩 System Architecture

```
             ┌──────────────────────┐
             │     User Uploads     │
             │   PDF / Image File   │
             └──────────┬───────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Pre-Processing     │
              │  (Text + Images)    │
              ├────────────────────┤
              │ PDF Parser          │
              │ BLIP Image Caption  │
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  Dynamic Prompt     │
              │  Library & Rules    │
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  LLM Inference API  │
              │ (OpenRouter / Cloud)│
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Compliance Checker  │
              │  Categorization     │
              └──────────┬─────────┘
                         │
                         ▼
              ┌────────────────────┐
              │   HITL Feedback UI  │
              │  Rule Reinforcement │
              └────────────────────┘
```

---

## 🧱 Core Components

| File | Description |
|------|--------------|
| `app.py` | Flask backend serving routes for upload, inference, and feedback. Manages user interactions and dynamic rule updates. |
| `Inference.py` | Main AI logic — handles BLIP image captioning, text extraction, prompt generation, and LLM inference. Integrates the prompt library dynamically. |
| `templates/` | HTML templates for UI (upload, settings, success, feedback pages). |
| `HitachiDS_Datathon_Challenges_Package/` | Sample documents used for compliance testing. |
| `requirements.txt` | Python dependencies for Flask, Transformers, Torch, and OpenAI/LLM APIs. |

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Compliance-Checker.git
cd Compliance-Checker
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment
Create a `.env` file at the project root:
```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 5️⃣ Run the App
```bash
python app.py
```
Then open your browser at **http://127.0.0.1:5000/**.

---

## 🧰 Example Usage

1. **Upload** a PDF file (memo, report, or form).
2. The system:
   - Extracts text and images.
   - Captions visuals using BLIP.
   - Combines textual and visual data.
   - Generates prompts dynamically from the prompt library.
3. The **LLM** processes the prompt tree and classifies the document.
4. The **UI displays results** with category and reasoning.
5. Users can:
   - Provide feedback on incorrect categories.
   - Add or modify rules directly from the interface.
   - Trigger re-evaluation with the new prompt logic.

---

## 🧱 Repository Structure

```
Datathon-backend/
├── app.py
├── Inference.py
├── requirements.txt
├── templates/
│   ├── upload.html
│   ├── settings.html
│   ├── success.html
│   ├── jobs.html
├── HitachiDS_Datathon_Challenges_Package/
│   ├── TC1_Sample_Public_Marketing_Document.pdf
│   ├── TC2_Filled_In_Employement_Application.pdf
│   ├── TC3_Sample_Internal_Memo.pdf
│   └── ...
└── .gitignore
```

---

## 🧠 Future Enhancements
- Add **RAG-based policy referencing** for grounding model outputs.
- Include **visual annotation of flagged content**.
- Enable **versioned prompt tracking** and audit trails.
- Expand HITL memory for **continual prompt refinement**.
- Integrate **multi-LLM support** for ensemble validation.

---

## 👥 Team & Credits
Developed by **Team SSD - TAMU Datathon 2025**  
Built for the **Hitachi Digital Services Datathon Challenge**  

---

## 📜 License
This project is licensed under the **MIT License**.
