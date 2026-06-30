# Agri-Subsidies AI Explorer: Knowledge Graph Mini-Project

## Overview
This project analyzes French state and regional subsidies (Agriculture, Ecology, and the Brittany Region) by leveraging Data Engineering, Knowledge Graphs (KG), and Artificial Intelligence. 

The pipeline ingests raw subsidy data, merges it with the official French company registry (Sirene), builds a semantic network of companies and funding sources, and uses logical inference to deduce specific statuses (e.g., MULTI_SOURCE_FUNDING, GREEN_TRANSITION). Finally, a PyKEEN TransE embedding model learns the vector representation of each company to recommend mathematically similar profiles.

---

## Project Structure

* **`POC.ipynb`**: The initial Jupyter Notebook used as a Proof of Concept (POC). It contains the preliminary data exploration, initial SIREN extraction logic, and early PyKEEN embedding tests to validate the feasibility of the Knowledge Graph.
* **`pipeline_phase1.py`**: The Data Engineering script. It dynamically loads and cleans CSV files from multiple sources (`agriculture`, `ecology`, `region`), handles different column names and separators, extracts the SIREN identifiers, and merges the data with the official Sirene database.
* **`pipeline_phase2_kg.py`**: The AI & Logic script. It builds the base Knowledge Graph (triplets), applies logical rules to infer new relationships (like AGRICULTURE_FUNDED or RECURRING_BENEFICIARY), and trains the PyKEEN TransE embedding model on the enriched graph.
* **`app.py`**: The interactive web interface built with Streamlit. It allows users to search for companies, filter the dataset by AI-inferred logical labels, view total subsidy amounts, and discover similar companies based on Cosine Similarity calculations.

---

## Requirements

To run this project, you need Python installed on your system along with the following libraries:

```bash
pip install pandas torch pykeen streamlit jupyter
```

*(Note: PyTorch (`torch`) will default to the CPU version, which is sufficient for this project, though a GPU can speed up Phase 2.)*

---

## How to Execute the Project

### Step 1: Data Preparation
Ensure your data is structured properly in the root directory before running the scripts, if it is not already done :
1. Place the massive Sirene dataset (`StockUniteLegale.csv`) in the root folder.
2. Create a `data_subsidies/` directory containing three sub-folders: `agriculture`, `ecology`, and `region`.
3. Place your raw `.csv` subsidy files into their respective sub-folders.

### Step 2: Run the Data Pipeline (Phase 1)
Open your terminal and run the first script to clean, standardize, and merge the data:
```bash
python Phase1.py
```
*Expected Output:* A merged dataset (`merged_subsidies_sirene.csv`) and a mapping dictionary (`siren_name_mapping.json`).

### Step 3: Build Graph and Train AI (Phase 2)
Run the second script to generate the triplets and train the Knowledge Graph Embeddings:
```bash
python Phase2.py
```
*Expected Output:* An enriched TSV graph (`knowledge_graph.tsv`) and a saved model directory (`pykeen_model/`). *Note: Training 50 epochs may take a few minutes.*

### Step 4: Launch the Web Application (Phase 3)
Start the Streamlit dashboard to interact with the AI model:
```bash
python -m streamlit run app.py
```
This will automatically open a local web page in your browser where you can explore the data and use the sidebar filters.