from datasets import load_dataset
from nltk import flatten
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from transformers import AutoTokenizer
from transformers import TFAutoModel
from vncorenlp import VnCoreNLP
import emoji
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regex as re
import tensorflow as tf


TRAIN_PATH = "dataset/train_dataset.csv"
VAL_PATH = "dataset/val_dataset.csv"
TEST_PATH = "dataset/test_dataset.csv"

raw_datasets = load_dataset(
    "csv", data_files={"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
)

print(raw_datasets)

df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)
df_test = pd.read_csv(TEST_PATH)


def make_outputs(data):
    outputs = []
    for row in range(len(data)):
        row_one_hot = []
        for col in range(1, len(data.columns)):
            sentiment = data.iloc[row, col]
            if sentiment == 0:
                one_hot = [1, 0, 0, 0]
            elif sentiment == 1:
                one_hot = [0, 1, 0, 0]
            elif sentiment == 2:
                one_hot = [0, 0, 1, 0]
            else:
                one_hot = [0, 0, 0, 1]
            row_one_hot.append(one_hot)
        outputs.append(row_one_hot)
    return np.array(outputs, dtype="uint8")


y_train = make_outputs(df_train)
y_val = make_outputs(df_val)
y_test = make_outputs(df_test)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


def convert_unicode(text):
    char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ"
    char1252 = char1252.split("|")
    charutf8 = charutf8.split("|")

    dic = {}
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dic[x.group()],
        text,
    )


# Standardize accent typing
vowels_to_ids = {}
vowels_table = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]

for i in range(len(vowels_table)):
    for j in range(len(vowels_table[i]) - 1):
        vowels_to_ids[vowels_table[i][j]] = (i, j)


def is_valid_vietnamese_word(word):
    chars = list(word)
    vowel_indexes = -1
    for index, char in enumerate(chars):
        x, y = vowels_to_ids.get(char, (-1, -1))
        if x != -1:
            if vowel_indexes == -1:
                vowel_indexes = index
            else:
                if index - vowel_indexes != 1:
                    return False
                vowel_indexes = index
    return True


def standardize_word_typing(word):
    if not is_valid_vietnamese_word(word):
        return word
    chars = list(word)
    dau_cau = 0
    vowel_indexes = []
    qu_or_gi = False

    for index, char in enumerate(chars):
        x, y = vowels_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True

        if y != 0:
            dau_cau = y
            chars[index] = vowels_table[x][0]

        if not qu_or_gi or index != 1:
            vowel_indexes.append(index)

    if len(vowel_indexes) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowels_to_ids.get(chars[1])
                chars[1] = vowels_table[x][dau_cau]
            else:
                x, y = vowels_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowels_table[x][dau_cau]
                else:
                    chars[1] = (
                        vowels_table[5][dau_cau]
                        if chars[1] == "i"
                        else vowels_table[9][dau_cau]
                    )
            return "".join(chars)
        return word

    for index in vowel_indexes:
        x, y = vowels_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowels_table[x][dau_cau]
            return "".join(chars)

    if len(vowel_indexes) == 2:
        if vowel_indexes[-1] == len(chars) - 1:
            x, y = vowels_to_ids[chars[vowel_indexes[0]]]
            chars[vowel_indexes[0]] = vowels_table[x][dau_cau]
        else:
            x, y = vowels_to_ids[chars[vowel_indexes[1]]]
            chars[vowel_indexes[1]] = vowels_table[x][dau_cau]
    else:
        x, y = vowels_to_ids[chars[vowel_indexes[1]]]
        chars[vowel_indexes[1]] = vowels_table[x][dau_cau]
    return "".join(chars)


def standardize_sentence_typing(text):
    words = text.lower().split()
    for index, word in enumerate(words):
        cw = re.sub(r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1/\2/\3", word).split("/")
        if len(cw) == 3:
            cw[1] = standardize_word_typing(cw[1])
        words[index] = "".join(cw)
    return " ".join(words)


replace_list = {
    "ô kêi": "ok",
    "okie": "ok",
    "o kê": "ok",
    "okey": "ok",
    "ôkê": "ok",
    "oki": "ok",
    "oke": "ok",
    "okay": "ok",
    "okê": "ok",
    "tks": "cảm ơn",
    "thks": "cảm ơn",
    "thanks": "cảm ơn",
    "ths": "cảm ơn",
    "thank": "cảm ơn",
    "kg": "không",
    "not": "không",
    "k": "không",
    "kh": "không",
    "kô": "không",
    "hok": "không",
    "ko": "không",
    "khong": "không",
    "kp": "không phải",
    "he he": "tích cực",
    "hehe": "tích cực",
    "hihi": "tích cực",
    "haha": "tích cực",
    "hjhj": "tích cực",
    "thick": "tích cực",
    "lol": "tiêu cực",
    "cc": "tiêu cực",
    "huhu": "tiêu cực",
    "cute": "dễ thương",
    "sz": "cỡ",
    "size": "cỡ",
    "wa": "quá",
    "wá": "quá",
    "qá": "quá",
    "đx": "được",
    "dk": "được",
    "dc": "được",
    "đk": "được",
    "đc": "được",
    "vs": "với",
    "vois": "với",
    "j": "gì",
    "“": " ",
    "time": "thời gian",
    "m": "mình",
    "mik": "mình",
    "r": "rồi",
    "bjo": "bao giờ",
    "very": "rất",
    "nhiu": "nhiêu",
    "dô": "vô",
    "h": "giờ",
    "coa": "có",
    "sv": "sinh viên",
    "gv": "giảng viên",
    "đ": "điểm",
    "tkb": "thời khóa biểu",
    "gud": "tốt",
    "wel done": "tốt",
    "good": "tốt",
    "gút": "tốt",
    "tot": "tốt",
    "nice": "tốt",
    "perfect": "rất tốt",
    "quality": "chất lượng",
    "chất lg": "chất lượng",
    "chat": "chất",
    "excellent": "hoàn hảo",
    "bt": "bài tập",
    "sad": "tệ",
    "por": "tệ",
    "poor": "tệ",
    "bad": "tệ",
    "beautiful": "đẹp tuyệt vời",
    "dep": "đẹp",
    "xau": "xấu",
    "sấu": "xấu",
    "thik": "thích",
    "iu": "yêu",
    "dt": "điện thoại",
    "fb": "facebook",
    "face": "facebook",
    "nt": "nhắn tin",
    "ib": "nhắn tin",
    "tl": "trả lời",
    "trl": "trả lời",
    "rep": "trả lời",
    "fback": "feedback",
    "fedback": "feedback",
    "sd": "sử dụng",
    "sài": "xài",
    "update": "cập nhật",
    "^_^": "tích cực",
    ":)": "tích cực",
    ":(": "tiêu cực",
    "❤️": "tích cực",
    "👍": "tích cực",
    "🎉": "tích cực",
    "😀": "tích cực",
    "😍": "tích cực",
    "😂": "tích cực",
    "🤗": "tích cực",
    "😙": "tích cực",
    "🙂": "tích cực",
    "😔": "tiêu cực",
    "😓": "tiêu cực",
    "⭐": "star",
    "*": "star",
    "🌟": "star",
}

with open("dataset/teencode.txt", encoding="utf-8") as f:
    for pair in f.readlines():
        key, value = pair.split("\t")
        replace_list[key] = value.strip()


def normalize_acronyms(text):
    words = []
    for word in text.strip().split():
        # word = word.strip(string.punctuation)
        if word.lower() not in replace_list.keys():
            words.append(word)
        else:
            words.append(replace_list[word.lower()])
    return emoji.demojize(" ".join(words))  # Remove Emojis


# Word segmentation
annotator = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar")


def word_segmentation(text):
    words = annotator.tokenize(text)
    return " ".join(word for word in flatten(words))


# Remove unnecessary characters
def remove_unnecessary_characters(text):
    text = re.sub(
        r"[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÍÌỈĨỊÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ_]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text


def text_preprocess(text):
    text = convert_unicode(text)
    text = standardize_sentence_typing(text)
    text = normalize_acronyms(text)
    text = word_segmentation(text)  # When use PhoBERT
    text = remove_unnecessary_characters(text)
    return text


tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
print(tokenizer.max_model_input_sizes)


def tokenize_function(dataset):
    clean_texts = list(map(text_preprocess, dataset["review"]))
    return tokenizer(
        clean_texts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print("input_ids of review 10:", tokenized_datasets["train"][10]["input_ids"])

MAX_SEQUENCE_LENGTH = tokenizer.model_max_length
MODEL_PATH = "./models"
BATCH_SIZE = 16

STEPS_PER_EPOCH = len(raw_datasets["train"]) // BATCH_SIZE
VALIDATION_STEPS = len(raw_datasets["val"]) // BATCH_SIZE
EPOCHS = 20

print(MAX_SEQUENCE_LENGTH)
print(STEPS_PER_EPOCH)
print(VALIDATION_STEPS)


def to_tensorflow_format(tokenized_dataset):
    features = tokenized_dataset.features
    return tokenized_dataset.remove_columns(list(features)[:-3]).with_format(
        "tensorflow"
    )


def preprocess_tokenized_dataset(
    tokenized_dataset, tokenizer, labels, batch_size, shuffle=False
):
    tf_dataset = to_tensorflow_format(tokenized_dataset)
    # features = {x: tf_dataset[x].to_tensor() for x in tokenizer.model_input_names}
    features = {x: tf_dataset[x] for x in tokenizer.model_input_names}
    labels = labels.reshape(len(labels), -1)

    tf_dataset = Dataset.from_tensor_slices((features, labels))
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(tf_dataset))
    return tf_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)


train_tf_dataset = preprocess_tokenized_dataset(
    tokenized_datasets["train"], tokenizer, y_train, BATCH_SIZE, shuffle=True
)
val_tf_dataset = preprocess_tokenized_dataset(
    tokenized_datasets["val"], tokenizer, y_val, BATCH_SIZE
)
test_tf_dataset = preprocess_tokenized_dataset(
    tokenized_datasets["test"], tokenizer, y_test, BATCH_SIZE
)

print(train_tf_dataset)


def create_model(optimizer):
    # https://riccardo-cantini.netlify.app/post/bert_text_classification
    inputs = {
        "input_ids": Input((MAX_SEQUENCE_LENGTH,), dtype="int32", name="input_ids"),
        "token_type_ids": Input(
            (MAX_SEQUENCE_LENGTH,), dtype="int32", name="token_type_ids"
        ),
        "attention_mask": Input(
            (MAX_SEQUENCE_LENGTH,), dtype="int32", name="attention_mask"
        ),
    }
    pretrained_bert = TFAutoModel.from_pretrained(
        "vinai/phobert-base", output_hidden_states=True
    )
    hidden_states = pretrained_bert(inputs).hidden_states

    # https://github.com/huggingface/transformers/issues/1328
    pooled_output = concatenate(
        tuple([hidden_states[i] for i in range(-4, 0)]),
        name="last_4_hidden_states",
        axis=-1,
    )[:, 0, :]
    x = Dropout(0.2)(pooled_output)
    print(pooled_output)

    outputs = concatenate(
        [
            Dense(
                units=4,
                activation="softmax",
                name=label,
            )(x)
            for label in df_train.columns[1:]
        ],
        axis=-1,
    )

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")
    return model


optimizer = Adam(learning_rate=1e-5)
type(optimizer)

# Stop if no improvement after 5 epochs
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

checkpoint_path = MODEL_PATH + "/checkpoints/cp-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every epoch
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    save_weights_only=True,
    save_freq=1 * STEPS_PER_EPOCH,
    verbose=1,
)

model = create_model(optimizer)

history = model.fit(
    train_tf_dataset,
    validation_data=val_tf_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stop_callback,
    ],
    verbose=1,
)

model.save_weights(f"{MODEL_PATH}/weights.h5")

fig = plt.figure(figsize=(15, 5))
plt.plot(
    history.history["loss"],
    linestyle="solid",
    marker="o",
    color="crimson",
    label="Train",
)
plt.plot(
    history.history["val_loss"],
    linestyle="solid",
    marker="o",
    color="dodgerblue",
    label="Validation",
)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss", fontsize=15)
plt.legend(loc="best")
fig.savefig(f"{MODEL_PATH}/evaluation.png", bbox_inches="tight")
plt.show()

reloaded_model = create_model(optimizer)
reloaded_model.load_weights(f"{MODEL_PATH}/weights.h5")
reloaded_model.summary()

plot_model(
    reloaded_model, to_file=f"{MODEL_PATH}/architecture.png", rankdir="LR", dpi=52
)

y_test_argmax = np.argmax(y_test, axis=-1)
print(y_test_argmax)

# Predict on test data


def predict(model, inputs, batch_size=1, verbose=0):
    y_pred = model.predict(inputs, batch_size=batch_size, verbose=verbose)
    y_pred = y_pred.reshape(len(y_pred), -1, 4)
    return np.argmax(y_pred, axis=-1)  # sentiment values (position that have max value)


def print_pred(replacements, aspects, sentence_pred):
    sentiments = map(lambda x: replacements[x], sentence_pred)
    for aspect, sentiment in zip(aspects, sentiments):
        if sentiment:
            print(f"=> {aspect},{sentiment}")


y_pred = predict(reloaded_model, test_tf_dataset, BATCH_SIZE, verbose=1)
reloaded_model.evaluate(test_tf_dataset, batch_size=BATCH_SIZE, verbose=1)

replacements = {0: None, 1: "negative", 2: "positive"}
aspects = df_test.columns[1:]

sen_idx = 19
print("Example:", df_test["review"][sen_idx])
print_pred(replacements, aspects, y_pred[sen_idx])

print("Report metrics")

print("Polarity Detection")

aspect_test = []
aspect_pred = []

for row_test, row_pred in zip(y_test_argmax, y_pred):
    for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
        aspect_test.append(bool(col_test) * aspects[index])
        aspect_pred.append(bool(col_pred) * aspects[index])

from sklearn.metrics import classification_report

aspect_report = classification_report(
    aspect_test, aspect_pred, digits=4, zero_division=1, output_dict=True
)
print(classification_report(aspect_test, aspect_pred, digits=4, zero_division=1))

print("Polarity Detection")

y_test_flat = y_test_argmax.flatten()
y_pred_flat = y_pred.flatten()
target_names = list(map(str, replacements.values()))

polarity_report = classification_report(
    y_test_flat, y_pred_flat, digits=4, output_dict=True
)
print(
    classification_report(y_test_flat, y_pred_flat, target_names=target_names, digits=4)
)

print("Aspect + Polarity")

aspect_polarity_test = []
aspect_polarity_pred = []

for row_test, row_pred in zip(y_test_argmax, y_pred):
    for index, (col_test, col_pred) in enumerate(zip(row_test, row_pred)):
        aspect_polarity_test.append(f"{aspects[index]},{replacements[col_test]}")
        aspect_polarity_pred.append(f"{aspects[index]},{replacements[col_pred]}")

aspect_polarity_report = classification_report(
    aspect_polarity_test,
    aspect_polarity_pred,
    digits=4,
    zero_division=1,
    output_dict=True,
)
print(
    classification_report(
        aspect_polarity_test, aspect_polarity_pred, digits=4, zero_division=1
    )
)

print("Summary")

aspect_dict = aspect_report["macro avg"]
aspect_dict["accuracy"] = aspect_report["accuracy"]

polarity_dict = polarity_report["macro avg"]
polarity_dict["accuracy"] = polarity_report["accuracy"]

aspect_polarity_dict = aspect_polarity_report["macro avg"]
aspect_polarity_dict["accuracy"] = aspect_polarity_report["accuracy"]

df_report = pd.DataFrame.from_dict([aspect_dict, polarity_dict, aspect_polarity_dict])
df_report.index = ["Aspect Detection", "Polarity Detection", "Aspect + Polarity"]
df_report.drop("support", axis=1)

print(df_report)

# Predict random text
while True:
    user_input = input("Enter your sentence or 'No' to quit: ")
    if user_input.lower() == "no":
        break
    example_input = text_preprocess(user_input)
    tokenized_input = tokenizer(example_input, padding="max_length", truncation=True)
    features = {x: [[tokenized_input[x]]] for x in tokenizer.model_input_names}

    pred = predict(reloaded_model, Dataset.from_tensor_slices(features))
    print_pred(replacements, aspects, pred[0])
