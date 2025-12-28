# CE49X - Introduction to Computational Thinking and Data Science for Civil Engineers
# Final Project: AI-Driven Civil Engineering News Analysis & Classification

**Students:** Berat Koncuk, Nur Bahar Gürsoy
**Instructor:** Dr. Eyuphan Koc
**Date:** December 27, 2025
**Institution:** Boğaziçi University, Department of Civil Engineering

---

## 📌 Project Overview
This project aims to automate the tracking of technological trends in the construction industry. By leveraging Web Scraping, Natural Language Processing (NLP), and Database Management systems, we developed an end-to-end pipeline that:

1.  **Scrapes** civil engineering news articles from various online sources (Global Construction Review, Construction Dive, etc.).
2.  **Summarizes** the content using AI-based text summarization techniques (LLM integration).
3.  **Classifies** articles into specific sub-domains (e.g., Geotechnical, Structural, Transportation) and identifies AI technologies mentioned (e.g., Computer Vision, Predictive Analytics).
4.  **Stores** the structured data in a PostgreSQL database (via Docker) for data integrity and future querying.

---

## 📂 File Structure & Deliverables

### 1. Main Analysis Code
* **`Final Project (Bahar&Berat).ipynb`**: The primary Jupyter Notebook containing the source code for data scraping, text preprocessing, summarization models, keyword tagging, and visualizations (Word Clouds, Heatmaps).

### 2. Database & Data Files
* **`proje_yedegi.sql` (SQL Dump)**: A complete export of the PostgreSQL database. This file allows for the reconstruction of the database schema and data (`final_tablo`) on any machine.
* **`Proje_Verileri.xlsx` (Excel Export)**: A user-friendly Excel export of the final database table. It contains the scraped articles, their AI-generated summaries, and assigned classification tags. *Recommended for quick review.*

### 3. Scripts & Configuration
* **`final_yukleme.py`**: The Python script used to merge the processed datasets (Cleaned Text + Tags) and upload them into the PostgreSQL database.
* **`requirements.txt`**: A list of Python libraries required to run the project environment.

---

## 🚀 How to View the Results

### Option A: Quick Review (No Installation Required)
For a direct view of the results without setting up a database environment, please refer to the **`Proje_Verileri.xlsx`** file. This file represents the exact state of the database table.

### Option B: Database Restoration (For Technical Review)
To replicate the database environment locally:
1.  Ensure **PostgreSQL** or **Docker** is installed.
2.  Import the **`proje_yedegi.sql`** file into your local instance using the command line or pgAdmin.
    * *Command Example:* `psql -U postgres -d postgres -f proje_yedegi.sql`
3.  The table **`final_tablo`** will be created with 553 entries.

---

## 🛠️ Methodology & Workflow

1.  **Data Collection:** Utilized `Selenium` and `BeautifulSoup` to scrape news articles related to "AI in Construction".
2.  **Preprocessing:** Applied NLP techniques (tokenization, stop-word removal) to clean raw text.
3.  **Tagging:** Articles were analyzed to detect specific Civil Engineering sub-fields and AI technologies using keyword matching.
4.  **Storage:** The final structured dataset was uploaded to a PostgreSQL container named `benim-db` to ensure data persistence.

---

## 📦 Requirements
To run the source code (`.ipynb` or `.py` files), the following Python libraries are required:

* `pandas`
* `numpy`
* `selenium`
* `beautifulsoup4`
* `sqlalchemy`
* `psycopg2-binary`
* `openpyxl`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `wordcloud`

---
*Fall 2025 - Boğaziçi University*
