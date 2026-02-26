import gradio as gr
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from groq import Groq
import os

# Load model
tokenizer = RobertaTokenizer.from_pretrained("mindscope_model")
model = RobertaForSequenceClassification.from_pretrained("mindscope_model")
model.eval()

# Groq client - paste your key here
client = Groq(api_key="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")

label_names = ['Depression 🍄', 'ADHD 💐', 'OCD 🐰', 'PTSD ☃️', 'Aspergers 🦀']
label_contexts = {
    'Depression 🍄': "the person seems to be experiencing feelings of depression or low mood",
    'ADHD 💐': "the person seems to be experiencing ADHD-like patterns of scattered focus and restlessness",
    'OCD 🐰': "the person seems to be experiencing OCD-like patterns of repetitive thoughts or checking behaviors",
    'PTSD ☃️': "the person seems to be experiencing PTSD-like patterns of hypervigilance or flashbacks",
    'Aspergers 🦀': "the person seems to be experiencing Aspergers-like patterns of social navigation and sensory sensitivity"
}

def get_groq_response(text, top_label):
    context = label_contexts.get(top_label, "the person shared something personal")
    prompt = f"""You are a warm, funny, gen-z therapist who gives ONE single short line of comfort. 
{context}. 
They wrote: "{text[:200]}"
Respond with exactly one line. Be warm, casual, slightly funny, never clinical. 
Do NOT mention diagnosis. Do NOT ask questions. Just one gentle human one-liner that makes them feel seen.
End with one emoji."""
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

def predict(text):
    if not text.strip():
        return {}, "✨ type something and i'll listen!"
    
    inputs = tokenizer(text, return_tensors="pt",
                      max_length=128, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).squeeze()
    result = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    
    top_label = max(result, key=result.get)
    groq_response = get_groq_response(text, top_label)
    
    return result, groq_response

css = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
body {
    background: linear-gradient(160deg, #fff0f5 0%, #ffecd2 40%, #f0f4ff 100%) !important;
    font-family: 'Nunito', sans-serif !important;
}
.gradio-container {
    background: rgba(255,255,255,0.92) !important;
    border-radius: 32px !important;
    max-width: 820px !important;
    margin: 40px auto !important;
    padding: 40px !important;
    box-shadow: 0 12px 40px rgba(253,121,168,0.12) !important;
}
.gr-button-primary {
    background: linear-gradient(90deg, #fd79a8, #fdcb6e) !important;
    border: none !important;
    border-radius: 50px !important;
    color: white !important;
    font-weight: 800 !important;
}
.gr-textbox textarea {
    border-radius: 20px !important;
    border: 2px solid #ffd6e7 !important;
    background: #fffaf9 !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}
footer {display: none !important;}
"""

description = """
<div style='text-align:center; font-family: Nunito, sans-serif; padding: 8px 0 24px 0;'>

<p style='font-size:17px; color:#636e72; max-width:640px; margin: 0 auto 16px auto; line-height:1.9;'>
So you too have just been talking to ChatGPT asking it for advice and feel rather <i>exhausted</i> about the way it responds? 'cause it's constantly asking so many questions? been there. 
instead, I bring you <b style='color:#e84393;'>MindScope</b>. 🌸
</p>

<p style='font-size:16px; color:#636e72; max-width:620px; margin: 0 auto 16px auto; line-height:1.8;'>
Instead of generic responses, this tool was trained on <b>real words from real people, </b>thousands of Reddit posts written by humans navigating depression, ADHD, OCD, PTSD, and Aspergers every single day.
</p>

<p style='font-size:16px; color:#636e72; max-width:620px; margin: 0 auto 16px auto; line-height:1.8;'>
Use this space to <b>journal freely</b>, paste something you wrote, or just type how you're feeling. MindScope doesn't judge, it listens, reads between the lines, and reflects back what it notices in your language. Built for <b>researchers, students, and the curious, </b>this is a window into how AI understands human emotion through language.
</p>

<div style='background: linear-gradient(90deg, #ffecd2, #ffd6e7); border-radius:16px; padding:16px 24px; max-width:580px; margin: 0 auto 16px auto;'>
<p style='font-size:14px; color:#636e72; margin:0; line-height:1.8;'>
💛 <i>A note from the person who built this:</i> sometimes a tub of ice cream helps too. This tool is <b>not a diagnosis</b>, not a replacement for therapy, and definitely not your best friend. It's just AI doing its best, like the rest of us. 🍦
</p>
</div>

<p style='font-size:15px; color:#e84393; font-weight:700; margin-top:8px;'>
🍽️ Plus hey, as a little token of appreciation and celebration, go break those virtual ceramic plates below! Thanks for trying this out! 
</p>

<p style='font-size:13px; color:#b2bec3;'>
⚠️ For research purposes only. Please seek professional support if you need it. 💙
</p>
</div>
"""

plates_html = """
<div style='text-align:center; padding: 24px; font-family: Nunito, sans-serif;'>
<p style='font-size:18px; font-weight:800; color:#e84393; margin-bottom:4px;'>🍽️ Celebratory Plate Breaking!</p>
<p style='font-size:14px; color:#636e72; margin-bottom:16px;'>Click a plate to break it! 🎉 (10 plates only, make them count!)</p>
<div id='plates' style='display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin-bottom:16px;'>
</div>
<p id='counter' style='font-size:14px; color:#a29bfe; font-weight:700;'>10 plates remaining 🍽️</p>
<p id='done-msg' style='font-size:16px; color:#e84393; font-weight:800; display:none;'>You did amazing today.</p>
<script>
(function() {
  const container = document.getElementById('plates');
  const counter = document.getElementById('counter');
  const doneMsg = document.getElementById('done-msg');
  let remaining = 10;
  
  for(let i = 0; i < 10; i++) {
    const plate = document.createElement('div');
    plate.innerHTML = '🍽️';
    plate.style.cssText = 'font-size:40px; cursor:pointer; transition: all 0.3s; user-select:none; filter:drop-shadow(0 4px 6px rgba(0,0,0,0.1));';
    plate.title = 'click to break!';
    plate.addEventListener('click', function() {
      if(this.innerHTML === '🍽️') {
        this.innerHTML = '✨';
        this.style.transform = 'scale(1.4) rotate(15deg)';
        this.style.opacity = '0.5';
        remaining--;
        if(remaining > 0) {
          counter.textContent = remaining + ' plates remaining 🍽️';
        } else {
          counter.style.display = 'none';
          doneMsg.style.display = 'block';
        }
      }
    });
    container.appendChild(plate);
  }
})();
</script>
</div>
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="yellow",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Nunito")
)) as demo:
    
    gr.HTML(f"<h1 style='text-align:center; font-size:42px; font-weight:900; color:#e84393; margin-bottom:0; font-family:Nunito,sans-serif;'>🌸 MindScope</h1>")
    gr.HTML(description)
    
    with gr.Row():
        text_input = gr.Textbox(
            lines=7,
            placeholder="💭 write anything here... a journal entry, how your day went, something on your mind. MindScope is listening.",
            label="your words 🌸"
        )
    
    submit_btn = gr.Button("✨ let MindScope read between the lines", variant="primary")
    
    with gr.Row():
        label_output = gr.Label(num_top_classes=5, label="🔍 what MindScope noticed")
        groq_output = gr.Textbox(label="💬 a little something for you", interactive=False)
    
    gr.Examples(
        examples=[
            ["I can't stop going back to check if I locked the door. I know I did but I just can't feel sure."],
            ["I haven't left my bed in three days. Nothing feels worth getting up for."],
            ["My brain never stops. I started four tasks today and finished none of them."],
            ["Loud noises make me freeze. I keep having flashbacks and don't know how to stop them."],
            ["I notice patterns in everything. Social situations drain me completely."],
        ],
        inputs=text_input
    )
    
    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_output, groq_output])
    
    gr.HTML("""
<div style='text-align:center; padding:24px; font-family:Nunito,sans-serif;'>
<p style='font-size:18px; font-weight:800; color:#e84393;'>🍽️ Celebratory Plate Breaking! opa!</p>
<p style='font-size:14px; color:#636e72; margin-bottom:16px;'>click a plate to break it 🎉</p>
<div style='display:flex; flex-wrap:wrap; justify-content:center; gap:16px;'>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
  <span onclick="this.innerHTML='💥'; this.style.fontSize='50px'" style='font-size:40px;cursor:pointer;'>🍽️</span>
</div>
<p style='font-size:13px; color:#b2bec3; margin-top:16px;'>10 plates · make them count! 🌸</p>
</div>
""")

demo.launch()