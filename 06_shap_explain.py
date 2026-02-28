import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

tokenizer = RobertaTokenizer.from_pretrained("mindscope_model")
model = RobertaForSequenceClassification.from_pretrained("mindscope_model")
model.eval()

label_names = ['Depression', 'ADHD', 'OCD', 'PTSD', 'Aspergers']

def predict_proba(texts):
    inputs = tokenizer(
        list(texts), return_tensors="pt",
        max_length=128, truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return F.softmax(outputs.logits, dim=1).numpy()

STOPWORDS = {
    'i', 'me', 'my', 'we', 'you', 'your', 'he', 'she', 'it', 'they',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'a', 'an',
    'the', 'and', 'but', 'or', 'if', 'of', 'at', 'by', 'for', 'with',
    'to', 'from', 'in', 'on', 'so', 'not', 'no', 'up', 'just', 'get',
    'got', 'like', 'know', 'want', 'go', 'going', 'really', 'now', 'also',
    'deleted', 'removed', 'edit', 'reddit', 'post', 'http', 'www', 'com',
    'one', 'even', 'back', 'still', 'more', 'very', 'much', 'than', 'then',
    'about', 'after', 'never', 'always', 'well', 'thing', 'things', 'as',
    'out', 'over', 'into', 'down', 'than', 'too', 'make', 'made', 'take',
    'that', 'this', 'these', 'those', 'what', 'which', 'who', 'when', 'how',
    'leted', 'don', 'ing', 'tion', 'ness', 'ment', 'ers', 'ting', 'ted',
    'ded', 'ing', 'hed', 'sed', 'ved', 'ned', 'red', 'led', 'ped',
    'que', 'ust', 'uch', 'ome', 'ike', 'ake', 'ive', 'ove', 'ave',
    'ere', 'ire', 'ore', 'ure', 'ase', 'ese', 'ose', 'use', 'ise', 'oved',
}

# Load 15 samples per class
test_df = pd.read_csv("test.csv")

for class_idx, class_name in enumerate(label_names):
    print(f"\nProcessing {class_name}...")

    samples = test_df[test_df['label'] == class_idx].sample(15, random_state=42)
    texts = [' '.join(row['body'].split()[:40]) for _, row in samples.iterrows()]
    sample_preview = [t[:70] + "..." for t in texts[:3]]

    explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
    shap_values = explainer(texts)

    # collect token:shap for this class across all 15 samples
    word_shap = {}
    for i in range(len(texts)):
        tokens = shap_values[i, :, class_idx].data
        vals = shap_values[i, :, class_idx].values
        for tok, val in zip(tokens, vals):
            # clean token
            clean = tok.replace("Ġ", "").replace("▁", "").strip().lower()
            # skip junk
            if not clean or len(clean) < 3:
                continue
            if clean in STOPWORDS:
                continue
            if clean in ["<s>", "</s>", "<pad>"]:
                continue
            if not clean.isalpha():
                continue
            # skip fragments that look like word endings
            if len(clean) < 4:
                continue
            # skip if it looks like a suffix fragment
            suffixes = ['ing', 'ted', 'led', 'ned', 'red', 'sed', 'ved', 
                        'ded', 'hed', 'ped', 'leted', 'tion', 'ness', 'ment']
            if any(clean == s for s in suffixes):
                continue
            if clean not in word_shap:
                word_shap[clean] = []
            word_shap[clean].append(val)

    # mean shap per word, min 3 occurrences
    word_means = {w: np.mean(v) for w, v in word_shap.items() if len(v) >= 3}

    if not word_means:
        print(f"  No words found for {class_name}, skipping")
        continue

    # top 15 by absolute mean
    sorted_words = sorted(word_means.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    words_plot = [w for w, _ in sorted_words]
    vals_plot = np.array([v for _, v in sorted_words])

    # sort by value
    sort_idx = np.argsort(vals_plot)
    words_plot = [words_plot[j] for j in sort_idx]
    vals_plot = vals_plot[sort_idx]

    colors = ['#E84393' if v > 0 else '#6B3A2A' for v in vals_plot]

    fig, (ax_main, ax_legend) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    ax_main.barh(
        range(len(words_plot)), vals_plot,
        color=colors, height=0.6,
        edgecolor='white', linewidth=0.5
    )
    ax_main.set_yticks(range(len(words_plot)))
    ax_main.set_yticklabels(words_plot, fontsize=11)
    ax_main.axvline(0, color='#333333', linewidth=1.2)
    ax_main.set_title(
        f'SHAP Global Summary {class_name}\nMean token impact across 15 real Reddit posts',
        fontsize=12, fontweight='bold', pad=16
    )
    ax_main.set_xlabel('Mean SHAP Value', fontsize=10, color='#555')
    ax_main.set_ylabel('Top Influential Words', fontsize=10, color='#555')
    ax_main.xaxis.grid(True, linestyle='--', alpha=0.4, color='#ccc')
    ax_main.set_axisbelow(True)
    ax_main.set_facecolor('#FAF7F4')
    fig.patch.set_facecolor('#FAF7F4')

    pink_patch = mpatches.Patch(color='#E84393', label='Pushes TOWARD this class')
    brown_patch = mpatches.Patch(color='#6B3A2A', label='Pushes AWAY from this class')
    ax_main.legend(handles=[pink_patch, brown_patch], loc='lower right', fontsize=9)

    # legend panel with sample sentences
    ax_legend.axis('off')
    legend_text = f"Sample posts ({class_name}):\n\n"
    for j, sent in enumerate(sample_preview):
        legend_text += f"{j+1}. {sent}\n\n"
    ax_legend.text(
        0.05, 0.95, legend_text,
        transform=ax_legend.transAxes,
        fontsize=7.5, verticalalignment='top', color='#555555',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff',
                  edgecolor='#DDD5C8', alpha=0.9)
    )

    plt.tight_layout()
    fname = f"shap_{class_name.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#FAF7F4')
    plt.close()
    print(f"  Saved {fname}")

print("\nAll SHAP plots saved!")