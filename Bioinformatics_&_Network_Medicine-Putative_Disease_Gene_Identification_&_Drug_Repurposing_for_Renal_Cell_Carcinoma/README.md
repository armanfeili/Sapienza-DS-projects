### 📌 **README.md – Putative Disease Gene Identification & Drug Repurposing for Renal Cell Carcinoma**  

# 🧬 Putative Disease Gene Identification & Drug Repurposing for Renal Cell Carcinoma  
**Bioinformatics & Network Medicine - Final Project**  
*Jan 2025 – Feb 2025*  

## 📖 Overview  
This project applies **Network Medicine & Bioinformatics** to identify **putative disease genes** and **repurpose existing drugs** for **Renal Cell Carcinoma (RCC)**. By leveraging **gene-disease associations (GDA), protein-protein interactions (PPI), and graph-based ranking algorithms**, we prioritized potential **disease genes** and mapped their **drug interactions** to identify existing treatments.  

## 🎯 Objectives  
✅ Identify **novel candidate genes** linked to RCC using **machine learning & network analysis**  
✅ Build a **disease-specific interactome** to model RCC-related molecular interactions  
✅ Rank disease-associated genes using **Random Walk with Restart (RWR), DIAMOnD, & DiaBLE** algorithms  
✅ Perform **gene enrichment analysis** (GO, KEGG, Reactome) to validate gene rankings  
✅ Investigate **drug-gene interactions** using **DGIdb** and **validate through clinical trials**  

## 🔬 Methodology  

1️⃣ **Interactome Reconstruction**  
   - Extract **RCC-specific genes** from **DISEASES** database  
   - Filter **human protein-protein interactions (PPI)** from **BioGRID**  
   - Construct a **graph-based disease interactome** and compute key **network metrics**  

2️⃣ **Disease Gene Identification**  
   - Implement **RWR, DIAMOnD, and DiaBLE** algorithms for gene prioritization  
   - Validate results using **5-fold cross-validation**  
   - Compare different algorithms using **Precision, Recall, F1-score, Accuracy**  

3️⃣ **Gene Enrichment Analysis**  
   - Perform **functional/pathway enrichment analysis** using **Enrichr API**  
   - Identify overrepresented **biological processes (GO), pathways (KEGG, Reactome)**  
   - Assess overlap between **known and predicted disease genes**  

4️⃣ **Drug Repurposing & Clinical Trials**  
   - Retrieve **drug-gene interactions** from **DGIdb**  
   - Rank drugs based on **target coverage** of top disease genes  
   - Search **clinical trials data (ClinicalTrials.gov)** for RCC-related drug repurposing evidence  

## 🛠️ Tools & Technologies  

- **Python** (Pandas, NumPy, SciPy, Scikit-learn, NetworkX, Matplotlib)  
- **Databases:** DISEASES, BioGRID, Enrichr, DGIdb, ClinicalTrials.gov  
- **Machine Learning:** RWR, DIAMOnD, DiaBLE algorithms  
- **Gene Enrichment Analysis:** GO, KEGG, Reactome (via Enrichr API)  

## 📂 Project Structure  

```bash
📂 Bioinformatics-Network-Medicine-Project  
│── 📂 datasets/                  # Raw data (GDA, PPI, DGIdb interactions, etc.)  
│── 📂 results/                   # Processed outputs, ranking tables, enrichment results  
│── 📂 scripts/                   # Python scripts for interactome building, ML, analysis  
│── ├── interactome.py            # Constructs the disease-specific interactome  
│── ├── gene_ranking.py           # Runs RWR, DIAMOnD, DiaBLE for gene prioritization  
│── ├── enrichment_analysis.py     # Performs GO/KEGG enrichment using Enrichr API  
│── ├── drug_repurposing.py        # Extracts drug-gene interactions from DGIdb  
│── ├── clinical_trials.py         # Fetches clinical trials data from ClinicalTrials.gov  
│── 📄 README.md                   # Project documentation  
│── 📄 requirements.txt             # Dependencies list  
│── 📄 LICENSE                      # License details  
```

## 🚀 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/Bioinformatics-Network-Medicine-Project.git
cd Bioinformatics-Network-Medicine-Project
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Scripts  
- **Build the Interactome:**  
  ```bash
  python scripts/interactome.py
  ```
- **Run Gene Prioritization Algorithms:**  
  ```bash
  python scripts/gene_ranking.py
  ```
- **Perform Gene Enrichment Analysis:**  
  ```bash
  python scripts/enrichment_analysis.py
  ```
- **Analyze Drug-Gene Interactions & Clinical Trials:**  
  ```bash
  python scripts/drug_repurposing.py
  python scripts/clinical_trials.py
  ```

## 📊 Results  

- **Top Ranked Putative RCC Genes:** Identified using **RWR, DIAMOnD, and DiaBLE**  
- **Key Enriched Pathways:** TP53 signaling, HIF-1 pathway, Apoptosis regulation  
- **Top Drug Candidates for RCC Treatment:** **Cisplatin, Dexamethasone, Hexachlorophene**  
- **Clinical Trials Validation:** **Cisplatin** linked to **46 RCC clinical trials**  

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
