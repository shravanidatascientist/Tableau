import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama


# ---------------------------
# LLM Insights
# ---------------------------
def generate_insights(df_summary):
    prompt = f"""
    Analyze the dataset summary and provide key insights:
    
    {df_summary}
    """

    response = ollama.chat(
        model="gemma2:2b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ---------------------------
# Visualizations
# ---------------------------
def generate_visualizations(df):
    plot_paths = []

    # Histograms
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    # Correlation Heatmap (outside loop)
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        heatmap_path = "correlation_heatmap.png"
        plt.savefig(heatmap_path)
        plot_paths.append(heatmap_path)
        plt.close()

    return plot_paths


# ---------------------------
# Main EDA Function
# ---------------------------
def eda_analysis(file):
    df = pd.read_csv(file.name)

    # Fill numeric missing values
    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    summary = df.describe(include='all').to_string()
    missing_values = df.isnull().sum().to_string()

    insights = generate_insights(summary)
    plots = generate_visualizations(df)

    report = f"""
Data Loaded Successfully

Summary:
{summary}

Missing Values:
{missing_values}

LLM Insights:
{insights}
"""

    return report, plots


# ---------------------------
# Gradio App
# ---------------------------
app = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(file_types=[".csv"]),
    outputs=[
        gr.Textbox(label="EDA REPORT"),
        gr.Gallery(label="Data Visualizations")
    ],
    title="LLM Powered EDA Data Analyzer",
    description="Upload any dataset and get AI-powered EDA insights."
)

app.launch(share=True)
        