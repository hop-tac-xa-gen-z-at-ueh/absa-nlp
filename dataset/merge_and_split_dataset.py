import pandas as pd

df = pd.read_csv("dataset/AWARE_Games.csv")
translated_df = pd.read_csv("dataset/translated_AWARE_Games_sentences.csv")
translated_df = translated_df.drop_duplicates(subset="sentence_id", keep="last")

grouped_df = df.groupby("sentence_id").size().reset_index(name="counts")
filtered_df = grouped_df[grouped_df["counts"] > 1]

merged_df = pd.merge(
    translated_df,
    df[["sentence_id", "category", "sentiment"]],
    on="sentence_id",
    how="left",
)
replace_dict = {"positive": 2, "negative": 1}
merged_df["sentiment"] = (
    merged_df["sentiment"].map(replace_dict).fillna(merged_df["sentiment"])
)

# count the number of samples for each category and sentiment
grouped_df = (
    merged_df.groupby(["category", "sentiment"]).size().reset_index(name="counts")
)
grouped_df["ten_percent"] = grouped_df["counts"].apply(lambda x: int(x * 0.1))

train_dataset = merged_df
# create test dataset
test_dataset = pd.DataFrame()

for _, group in grouped_df.iterrows():
    category = group["category"]
    sentiment = group["sentiment"]
    sample_size = group["ten_percent"]

    sample_df = train_dataset[
        (train_dataset["category"] == category)
        & (train_dataset["sentiment"] == sentiment)
    ].sample(n=sample_size, random_state=42)
    test_dataset = pd.concat([test_dataset, sample_df])

test_dataset = test_dataset.drop_duplicates(subset="sentence_id")
train_dataset = train_dataset.drop(test_dataset.index)

# create validation dataset
val_dataset = pd.DataFrame()

for _, group in grouped_df.iterrows():
    category = group["category"]
    sentiment = group["sentiment"]
    sample_size = group["ten_percent"]

    sample_df = train_dataset[
        (train_dataset["category"] == category)
        & (train_dataset["sentiment"] == sentiment)
    ].sample(n=sample_size, random_state=42)
    val_dataset = pd.concat([val_dataset, sample_df])

val_dataset = val_dataset.drop_duplicates(subset="sentence_id")
train_dataset = train_dataset.drop(val_dataset.index)

pivot_df = merged_df.pivot(
    index="sentence_id", columns="category", values="sentiment"
).reset_index()
pivot_df = pivot_df.fillna(0)
pivot_df.reset_index(drop=True, inplace=True)
pivot_df = pd.merge(
    pivot_df,
    translated_df[["sentence_id", "translated_sentence"]],
    on="sentence_id",
    how="inner",
)

# Reorder the columns
cols = list(pivot_df.columns)
cols.remove("sentence_id")
cols.remove("translated_sentence")
pivot_df = pivot_df[["sentence_id", "translated_sentence"] + cols]
pivot_df = pivot_df.rename(columns={"translated_sentence": "review"})

# extract pivot dataset for train, val, and test
test_dataset_pivot = pivot_df[pivot_df["sentence_id"].isin(test_dataset["sentence_id"])]
pivot_df = pivot_df[~pivot_df["sentence_id"].isin(test_dataset["sentence_id"])]

val_dataset_pivot = pivot_df[pivot_df["sentence_id"].isin(val_dataset["sentence_id"])]
pivot_df = pivot_df[~pivot_df["sentence_id"].isin(val_dataset["sentence_id"])]

train_dataset_pivot = pivot_df

train_dataset_pivot = train_dataset_pivot.drop("sentence_id", axis=1)
val_dataset_pivot = val_dataset_pivot.drop("sentence_id", axis=1)
test_dataset_pivot = test_dataset_pivot.drop("sentence_id", axis=1)

test_dataset_pivot.to_csv("dataset/test_dataset.csv", index=False)
val_dataset_pivot.to_csv("dataset/val_dataset.csv", index=False)
train_dataset_pivot.to_csv("dataset/train_dataset.csv", index=False)
