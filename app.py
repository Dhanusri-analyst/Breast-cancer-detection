
import gradio as gr
import joblib
import pandas as pd

bundle = joblib.load("model/breast_cancer_model.joblib")
pipeline, threshold, features = bundle["pipeline"], bundle["threshold"], bundle["features"]

data = pd.read_csv("cancer prediction data.csv")   # used only for default values and examples
medians = data[features].median()


def predict(*values):
    row = pd.DataFrame([values], columns=features)
    p = pipeline.predict_proba(row)[0, 1]
    label = "MALIGNANT (suspicious: needs clinical follow-up)" if p >= threshold else "BENIGN"
    return label, f"{p:.1%}"


def load_example(kind):
    sample = data[data["diagnosis"] == kind].sample(1)[features].iloc[0]
    return [float(v) for v in sample]


groups = {
    "Mean values": [f for f in features if f.endswith("_mean")],
    "Standard error": [f for f in features if f.endswith("_se")],
    "Worst values": [f for f in features if f.endswith("_worst")],
}

with gr.Blocks(title="Breast Cancer Detection") as demo:
    gr.Markdown(
        "# Breast Cancer Detection\n"
        "Enter the 30 nucleus measurements, or load an example, then press **Predict**. "
        "This is a student ML project and **not a medical device**; never use it for real diagnosis."
    )

    components = {}
    with gr.Tabs():
        for group_name, cols in groups.items():
            with gr.Tab(group_name):
                for i in range(0, len(cols), 5):
                    with gr.Row():
                        for f in cols[i:i + 5]:
                            components[f] = gr.Number(label=f, value=float(medians[f]))
    inputs = [components[f] for f in features]   # same order the model was trained on

    with gr.Row():
        btn_m = gr.Button("Load a malignant example")
        btn_b = gr.Button("Load a benign example")
    with gr.Row():
        out_label = gr.Textbox(label="Prediction")
        out_prob = gr.Textbox(label="Estimated probability of malignant")
    submit = gr.Button("Predict", variant="primary")

    btn_m.click(lambda: load_example("M"), outputs=inputs)
    btn_b.click(lambda: load_example("B"), outputs=inputs)
    submit.click(predict, inputs=inputs, outputs=[out_label, out_prob])

import os
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))