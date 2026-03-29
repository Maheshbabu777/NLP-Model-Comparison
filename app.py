import gradio as gr
from transformers import pipeline

bert_model = pipeline("sentiment-analysis")
distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def compare_models(text):
    result1 = bert_model(text)[0]
    result2 = distilbert_model(text)[0]

    return(
        f"BERT -> {result1['label']} {round(result1['score'],2)}\n"
        f"DistilBERT -> {result2['label']} {round(result2['score'],2)}"
    )

iface = gr.Interface(
    fn=compare_models,
    inputs=gr.Textbox(label="Enter Text"),
    outputs="text",
    title="NLP Model Comparision",
    description="Compare predictions of multiple NLP models"
)

iface.launch()