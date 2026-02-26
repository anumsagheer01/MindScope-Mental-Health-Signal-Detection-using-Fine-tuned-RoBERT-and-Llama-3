from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = load_dataset("solomonk/reddit_mental_health_posts")
df = pd.DataFrame(dataset['train'])

# Keep only what we need
df = df[['body', 'subreddit']].dropna()
df = df[df['body'].str.strip() != '']

print("Subreddits found:", df['subreddit'].unique())
print("Counts:", df['subreddit'].value_counts())

# Create numeric label
label2id = {'depression': 0, 'ADHD': 1, 'OCD': 2, 'ptsd': 3, 'aspergers': 4}
df['label'] = df['subreddit'].map(label2id)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Sample evenly
samples = []
for sub in label2id.keys():
    subset = df[df['subreddit'] == sub]
    samples.append(subset.sample(min(5000, len(subset)), random_state=42))
df = pd.concat(samples).reset_index(drop=True)

# Split
train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
print("Done!")