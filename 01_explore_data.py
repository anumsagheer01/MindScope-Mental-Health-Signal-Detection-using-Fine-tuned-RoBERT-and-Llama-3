from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = load_dataset("solomonk/reddit_mental_health_posts")
df = pd.DataFrame(dataset['train'])

print(df['subreddit'].value_counts())
print(df.shape)

# Visualize
plt.figure(figsize=(10,6))
sns.countplot(y='subreddit', data=df, order=df['subreddit'].value_counts().index)
plt.title("Posts per Subreddit")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()