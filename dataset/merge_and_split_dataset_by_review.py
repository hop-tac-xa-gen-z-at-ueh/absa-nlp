import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/AWARE_Games.csv")
translated_df = pd.read_csv("dataset/translated_AWARE_Games_sentences.csv")
translated_df = translated_df.drop_duplicates(subset="sentence_id", keep="last")

review_df = pd.merge(
    translated_df,
    df[
        [
            "review_id",
            "review",
            "sentence_id",
            "sentence",
        ]
    ],
    on="sentence_id",
).drop_duplicates(subset=["review_id", "sentence_id"], keep="last")
review_df["sentence_position"] = review_df.apply(
    lambda row: row["review"].find(row["sentence"]), axis=1
)
review_df = (
    review_df.sort_values(by="sentence_position")
    .groupby(["review_id"])["translated_sentence"]
    .apply(" ".join)
    .reset_index(name="review")
)

review_polarity_df = (
    df.groupby(["review_id", "category", "sentiment"])
    .size()
    .reset_index(name="counts")
    .sort_values(by=["category", "sentiment", "counts"], ascending=[True, False, True])
)
review_polarity_df = review_polarity_df.drop_duplicates(
    subset=["review_id", "category"], keep="last"
)
replace_dict = {"positive": 2, "negative": 1}
review_polarity_df["sentiment"] = (
    review_polarity_df["sentiment"]
    .map(replace_dict)
    .fillna(review_polarity_df["sentiment"])
)

pivot_df = review_polarity_df.pivot(
    index="review_id", columns="category", values="sentiment"
).reset_index()
pivot_df = pivot_df.fillna(0)
pivot_df.reset_index(drop=True, inplace=True)
pivot_df = pd.merge(
    pivot_df,
    review_df[["review_id", "review"]],
    on="review_id",
    how="inner",
)
pivot_df = pivot_df.drop(columns=["review_id"])

# # Reorder the columns
cols = list(pivot_df.columns)
cols.remove("review")
pivot_df = pivot_df[["review"] + cols]

print(f"Total rows: {len(pivot_df)}")
# Calculate 10% of total rows
test_size = val_size = int(len(pivot_df) * 0.1)

# Create a temporary train set and a test set
temp_train_dataset_pivot, test_dataset_pivot = train_test_split(
    pivot_df, test_size=test_size, random_state=42
)

# Split the temporary train set into the final train set and a validation set
train_dataset_pivot, val_dataset_pivot = train_test_split(
    temp_train_dataset_pivot,
    test_size=val_size / (len(pivot_df) - test_size),
    random_state=42,
)

# Print the sizes of the train, validation, and test sets
print(f"Train set size: {len(train_dataset_pivot)}")
print(f"Validation set size: {len(val_dataset_pivot)}")
print(f"Test set size: {len(test_dataset_pivot)}")

test_dataset_pivot.to_csv("dataset/test_dataset.csv", index=False)
val_dataset_pivot.to_csv("dataset/val_dataset.csv", index=False)
train_dataset_pivot.to_csv("dataset/train_dataset.csv", index=False)
