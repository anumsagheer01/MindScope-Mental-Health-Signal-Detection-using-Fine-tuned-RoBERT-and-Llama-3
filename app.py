import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from groq import Groq
import os

# Load model
tokenizer = RobertaTokenizer.from_pretrained("anumsagheer/mindscope-model")
model = RobertaForSequenceClassification.from_pretrained("anumsagheer/mindscope-model")
model.eval()

<<<<<<< HEAD
# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
=======
<<<<<<< HEAD
# Groq client - paste your key here
client = Groq(api_key="GROQ_API_KEY")
=======
# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
>>>>>>> a17223a (update UI, add SHAP explainability plots)
>>>>>>> fa8a085526fe0caaac22ea4e97f2104e50dce2dc

label_names = ['Depression', 'ADHD', 'OCD', 'PTSD', 'Aspergers']
label_contexts = {
    'Depression': "the person seems to be experiencing feelings of depression or low mood",
    'ADHD': "the person seems to be experiencing ADHD-like patterns of scattered focus and restlessness",
    'OCD': "the person seems to be experiencing OCD-like patterns of repetitive thoughts or checking behaviors",
    'PTSD': "the person seems to be experiencing PTSD-like patterns of hypervigilance or flashbacks",
    'Aspergers': "the person seems to be experiencing Aspergers-like patterns of social navigation and sensory sensitivity"
}

def get_groq_response(text, top_label):
    context = label_contexts.get(top_label, "the person shared something personal")
    prompt = f"""You are a warm, thoughtful AI that gives ONE single short line of acknowledgment.
{context}.
They wrote: "{text[:200]}"
Respond with exactly one line. Be warm, calm, human. Never clinical. Never ask questions. Never mention diagnosis.
Just one gentle sentence that makes them feel heard. End with one subtle emoji."""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

def predict(text):
    if not text.strip():
        return {}, "Type something above and MindScope will analyze it."

    inputs = tokenizer(text, return_tensors="pt",
                       max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    result = {label_names[i]: float(probs[i]) for i in range(len(label_names))}

    top_label = max(result, key=result.get)
    top_confidence = max(result.values())

    if top_confidence < 0.50:
        return result, "No strong signal detected. Your words don't point clearly in any one direction. 🌿"

    groq_response = get_groq_response(text, top_label)
    return result, groq_response

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body {
    background-color: #F5F0EA !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.gradio-container {
    background-color: #F5F0EA !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 60px 40px !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* Input textbox */
.gr-textbox textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #DDD5C8 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    color: #2C2420 !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    resize: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

.gr-textbox textarea:focus {
    border-color: #8B4513 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(139,69,19,0.08) !important;
}

/* Button */
button.primary {
    background-color: #6B3A2A !important;
    color: #FAF7F4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease !important;
}

button.primary:hover {
    background-color: #8B4513 !important;
}

/* Output boxes */
.gr-label, .gr-textbox {
    background-color: #FFFFFF !important;
    border: 1px solid #DDD5C8 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* Labels */
label, .gr-label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #8A7668 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

footer { display: none !important; }
"""

header_html = """
<div style='padding: 0 0 48px 0; font-family: Inter, -apple-system, sans-serif;'>
    <p style='font-size: 11px; font-weight: 600; letter-spacing: 2px; color: #8A7668; text-transform: uppercase; margin: 0 0 16px 0;'>Research Tool</p>
    <h1 style='font-size: 38px; font-weight: 300; color: #2C2420; margin: 0 0 8px 0; letter-spacing: -0.5px;'>MindScope</h1>
    <p style='font-size: 15px; color: #6B5C52; font-weight: 400; margin: 0 0 24px 0; line-height: 1.6;'>
        A fine-tuned RoBERTa model that reads text and identifies mental health language patterns across five categories.
        Trained on 150,000 real Reddit posts. Powered by Llama 3 for contextual responses.
    </p>
    <div style='width: 40px; height: 2px; background-color: #8B4513; border-radius: 2px;'></div>
</div>
"""

disclaimer_html = """
<div style='margin-top: 48px; padding-top: 24px; border-top: 1px solid #DDD5C8; font-family: Inter, sans-serif;'>
    <p style='font-size: 12px; color: #A0907E; line-height: 1.7; margin: 0;'>
        <strong style='color: #6B5C52;'>Research purposes only.</strong> 
        MindScope is not a diagnostic tool and is not a replacement for professional mental health support. 
    </p>
</div>
"""

with gr.Blocks(
    css=css,
    theme=gr.themes.Base(
        font=gr.themes.GoogleFont("Inter"),
        primary_hue="orange",
        neutral_hue="stone",
    )
) as demo:
<<<<<<< HEAD

    gr.HTML(header_html)

    text_input = gr.Textbox(
        lines=6,
        placeholder="Write anything here, whatever's on your mind",
        label="Input",
        show_label=True,
    )

    submit_btn = gr.Button("Analyze", variant="primary", size="sm")

    with gr.Row():
        label_output = gr.Label(
            num_top_classes=5,
            label="Signal Detection"
        )
        groq_output = gr.Textbox(
            label="Response",
            interactive=False,
            lines=4,
        )

    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_output, groq_output])

    gr.HTML(disclaimer_html)
=======
>>>>>>> fa8a085526fe0caaac22ea4e97f2104e50dce2dc

    gr.HTML(header_html)

    text_input = gr.Textbox(
        lines=6,
        placeholder="Write anything here, whatever's on your mind",
        label="Input",
        show_label=True,
    )

    submit_btn = gr.Button("Analyze", variant="primary", size="sm")

    with gr.Row():
        label_output = gr.Label(
            num_top_classes=5,
            label="Signal Detection"
        )
        groq_output = gr.Textbox(
            label="Response",
            interactive=False,
            lines=4,
        )

    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_output, groq_output])

    gr.HTML(disclaimer_html)

demo.launch()
