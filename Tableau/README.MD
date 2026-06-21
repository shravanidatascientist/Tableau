# 📊 AI & Data Science Job Market Analysis 2020–2024
### Interactive Tableau Dashboard | Python Data Analysis

![Dashboard Preview](dashboard_preview.png)

> **"Which AI roles pay the most? Where are the jobs? Is remote work the norm?"**
> This dashboard answers all of it — with real data.

---

## 🎯 Business Problem

The AI and Data Science job market is growing at an unprecedented rate. However, professionals, students, and organizations often lack clear visibility into:

- Which AI/Data Science roles pay the highest salaries globally?
- How does experience level impact compensation?
- Which countries are the top hiring hubs for AI talent?
- Is remote work becoming the norm in Data Science?
- How has the AI job market grown from 2020 to 2024?

---

## 🏆 Goal

To perform an end-to-end data analysis on **3,755 global AI & Data Science job records** and build an interactive Tableau dashboard that delivers actionable insights for:

- 🎯 **Job seekers** — understand which roles and locations offer the best pay
- 🏢 **Recruiters** — benchmark salaries across experience levels and company sizes
- 📈 **Organizations** — understand global AI hiring trends and remote work patterns

---

## 📌 Key Insights Discovered

| # | Insight |
|---|---------|
| 1 | 🇺🇸 **USA dominates AI hiring** — accounts for the majority of high-paying AI jobs globally |
| 2 | 💰 **Executive roles earn $194,931** avg vs Entry Level at $78,546 — a **2.5x gap** |
| 3 | 🚀 **AI job market grew 2000%+** from just 76 jobs in 2020 to 1,785 in 2023 |
| 4 | 🏠 **Remote & Hybrid = 60%+** of all AI roles — remote work is mainstream in AI |
| 5 | 🤖 **Data Science Tech Lead** commands the highest average salary globally |
| 6 | 📊 **Average AI salary = $137,570** across 72 countries analyzed |

---

## 📊 Dashboard Features

| Chart | Insight |
|-------|---------|
| 🗺️ World Map | Average AI salary by country — color coded by pay level |
| 📊 Bar Chart | Top 10 highest paying AI & Data Science job titles |
| 📉 Bar Chart | Salary progression from Entry → Mid → Senior → Executive |
| 📈 Line Chart | AI job market growth trend 2020 to 2024 |
| 🍩 Donut Chart | Remote vs Hybrid vs On-Site work mode split |
| 🔢 KPI Tiles | Total jobs analyzed · Average salary · Countries covered |

**✨ Interactive filters:** Filter by Experience Level and Work Year — all charts update simultaneously!

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Data cleaning & EDA |
| **Pandas** | Data manipulation & transformation |
| **NumPy** | Numerical operations |
| **Tableau Public** | Interactive dashboard creation |
| **Kaggle** | Dataset source |

---

## 📁 Project Structure

```
📦 AI-JobMarket-Tableau-Dashboard
 ┣ 📜 README.md
 ┣ 📓 AI_JobMarket_Analysis.ipynb       ← Jupyter notebook (cleaning + EDA)
 ┣ 📄 ds_salaries.csv                   ← Raw dataset
 ┣ 📄 ds_salaries_clean.csv             ← Cleaned dataset (Tableau input)
 ┗ 📸 dashboard_preview.png             ← Dashboard screenshot
```

---

## 🔄 Data Cleaning Steps

```python
# 1. Loaded raw dataset — 3,755 rows, 11 columns
df = pd.read_csv('ds_salaries.csv')

# 2. Replaced short codes with readable labels
df['experience_level'] = df['experience_level'].replace({
    'EN': 'Entry Level', 'MI': 'Mid Level',
    'SE': 'Senior Level', 'EX': 'Executive'
})
df['remote_ratio'] = df['remote_ratio'].replace({
    0: 'On-Site', 50: 'Hybrid', 100: 'Remote'
})
df['company_size'] = df['company_size'].replace({
    'S': 'Small', 'M': 'Medium', 'L': 'Large'
})

# 3. Dropped unnecessary columns (salary, salary_currency)
# 4. Renamed all columns to clean readable names
# 5. Removed duplicates and salary outliers
# 6. Exported clean CSV for Tableau
df.to_csv('ds_salaries_clean.csv', index=False)
```

---

## 📥 Dataset

- **Source:** [Kaggle — AI/ML/Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
- **Records:** 3,755 job records
- **Years:** 2020 – 2024
- **Countries:** 72 countries
- **Features:** Job title, experience level, salary (USD), company location, remote ratio, company size

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/shravanidatascientist/AI-JobMarket-Tableau-Dashboard

# 2. Install dependencies
pip install pandas numpy matplotlib

# 3. Open Jupyter Notebook
jupyter notebook AI_JobMarket_Analysis.ipynb

# 4. Run all cells to generate ds_salaries_clean.csv

# 5. Open ds_salaries_clean.csv in Tableau Public to explore dashboard
```

---

## 📈 EDA Highlights

```
Dataset Shape    : 3,755 rows × 9 columns (after cleaning)
Null Values      : 0 (clean dataset)
Avg Salary (USD) : $137,570
Salary Range     : $30,000 – $400,000
Top Country      : United States
Top Job Title    : Data Science Tech Lead
Fastest Growth   : 2021 → 2022 (623% increase in job postings)
```

---

## 👩‍💻 About Me

**Bojja Sravani** — Agentic AI Engineer | Data Scientist

Passionate about building intelligent AI systems and extracting insights from data. Currently building production-grade Agentic AI applications using LangChain, CrewAI, LangGraph, and n8n.

🔗 **LinkedIn:** [linkedin.com/in/shravani-bojja-6b4b5a205](https://linkedin.com/in/shravani-bojja-6b4b5a205)
🐙 **GitHub:** [github.com/shravanidatascientist](https://github.com/shravanidatascientist)
📧 **Email:** shravanibojja71@gmail.com

---

## ⭐ If this project helped you, give it a star!

*Built with 💙 by Bojja Shravani*
