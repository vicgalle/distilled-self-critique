import requests
import json

import tqdm

from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="DataMonke/distilbert-base-uncased-sentiment-analysis-movie-reviews",
)


def generate(prompt, context=None):
    data = {
        "model": "stable-beluga:7b",  # change to "stable-beluga:13b" for the other model size.
        "prompt": prompt,
        "stream": False,
    }

    if context:
        data["context"] = context

    response = requests.post(
        "http://localhost:11434/api/generate", data=json.dumps(data)
    ).json()

    return response["response"], response["context"]


def compute_metrics(text):
    pred = int(pipe(text)[0]["label"][6]) + 1  # score from 1 to 5
    return pred


with open("./movies.txt", "r") as f:
    content = f.readlines()


movies = [m.strip() for m in content]

prompts = [
    f"Generate a negative review about the movie {m}"
    if i % 2
    else f"Generate a review about the movie {m}. Be negative"
    for i, m in enumerate(movies)
]

data = [{"prompt": p} for p in prompts]

name = "sentiment_review"

results = []

for sample in tqdm.tqdm(data):
    prompt = sample["prompt"]
    summary_pre, context = generate(prompt)

    critique, context = generate(
        "Identify specific ways in which the previous response could be made more positive",
        context,
    )

    summary_after, _ = generate(
        "Please, rewrite your original response using the previous critique to make the sentiment more positive",
        context,
    )

    try:
        scores_pre, scores_after = compute_metrics(summary_pre), compute_metrics(
            summary_after
        )
    except:
        continue

    results.append(
        {
            "summary_pre": summary_pre,
            "summary_after": summary_after,
            "scores_pre": scores_pre,
            "scores_after": scores_after,
            "prompt": prompt,
        }
    )

    with open(f"results_{name}.json", "w") as f:
        json.dump(results, f)
