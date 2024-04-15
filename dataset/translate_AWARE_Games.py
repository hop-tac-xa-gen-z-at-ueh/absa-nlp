import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def translate_to_vi(game_title, sentence):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"You will be provided with a sentence extracted from a review of the game {game_title} in English, and your task is to translate it into Vietnamese.",
            },
            {"role": "user", "content": sentence},
        ],
        temperature=0.7,
        top_p=1,
    )
    return response.choices[0].message.content


df = pd.read_csv("dataset/AWARE_Games.csv")
translated_df = pd.DataFrame(columns=["sentence_id", "translated_sentence"])

for index, row in df.iterrows():
    print(row["sentence_id"])
    print(row["sentence"])
    sentence_vi = translate_to_vi(row["app"], row["sentence"])
    new_row = pd.DataFrame(
        {"sentence_id": [row["sentence_id"]], "translated_sentence": [sentence_vi]}
    )
    translated_df = pd.concat([translated_df, new_row], ignore_index=True)
    translated_df.to_csv("dataset/translated_AWARE_Games_sentences.csv", index=False)
