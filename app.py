"""Проверка двух моделей на CPU в Streamlit Community Cloud."""

import hashlib
import json
import logging
import re
import resource
import unicodedata
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from streamlit.errors import StreamlitSecretNotFoundError


ENCODER_ID = "intfloat/multilingual-e5-small"
MODEL_FILE = "tfidf_logreg_5class_v2.joblib"
MODEL_INFO = "model_info_v2.json"
MAX_CHARS = 3_000
MAX_TOKENS = 128
BATCH_SIZE = 8
POLARITY_CLASSES = {
    "Теріс": ("very_negative", "negative"),
    "Бейтарап": ("neutral",),
    "Оң": ("positive", "very_positive"),
}


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@st.cache_resource(show_spinner=False)
def load_models(repo_id, revision):
    # E5-small использует архитектуру BERT и токенизатор XLM-R.
    from transformers.models.bert.modeling_bert import BertModel
    from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

    # Указывайте только свой репозиторий с артефактом из ноутбука 02b.
    info_path = Path(hf_hub_download(repo_id, MODEL_INFO, revision=revision))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected = info["sklearn_version"]
    if sklearn.__version__ != expected:
        raise ValueError(f"В requirements.txt поставь scikit-learn=={expected} "
                         f"(установлена версия {sklearn.__version__}).")

    model_path = Path(hf_hub_download(repo_id, MODEL_FILE, revision=revision))
    with model_path.open("rb") as model_file:
        actual_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual_sha256 != info["model_sha256"]:
        raise ValueError("SHA-256 файла модели не совпал с model_info_v2.json.")
    sentiment_model = joblib.load(model_path)

    torch.set_num_threads(2)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(ENCODER_ID)
    encoder = BertModel.from_pretrained(ENCODER_ID, use_safetensors=True).eval()
    return sentiment_model, tokenizer, encoder


def predict_polarity(model, text):
    class_scores = dict(zip(model.classes_, model.predict_proba([text])[0]))
    scores = {
        polarity: sum(class_scores[label] for label in labels)
        for polarity, labels in POLARITY_CLASSES.items()
    }
    return max(scores, key=scores.__getitem__)


def make_chunks(tokenizer, text):
    text_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    prefix_ids = tokenizer.encode("query:", add_special_tokens=False)
    chunk_size = MAX_TOKENS - len(prefix_ids) - tokenizer.num_special_tokens_to_add(pair=False)
    chunks = []
    for start in range(0, len(text_ids), chunk_size):
        input_ids = prefix_ids + text_ids[start : start + chunk_size]
        chunks.append({"input_ids": tokenizer.build_inputs_with_special_tokens(input_ids)})
    return chunks, len(text_ids)


def check_encoder(tokenizer, encoder, text):
    chunks, text_tokens = make_chunks(tokenizer, text)
    if not chunks:
        raise ValueError("После токенизации не осталось текста для анализа.")

    with torch.inference_mode():
        for start in range(0, len(chunks), BATCH_SIZE):
            inputs = tokenizer.pad(
                chunks[start : start + BATCH_SIZE], padding=True,
                return_attention_mask=True, return_tensors="pt",
            )
            hidden = encoder(**inputs).last_hidden_state
            attention = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1)
            if not torch.isfinite(pooled).all().item():
                raise ValueError("E5 вернул некорректные значения эмбеддингов.")

    max_chunk_tokens = max(len(chunk["input_ids"]) for chunk in chunks)
    return text_tokens, len(chunks), max_chunk_tokens


def main():
    st.set_page_config(page_title="Kazakh NLP Analyzer · Шаг 0", layout="centered")
    st.title("Kazakh NLP Analyzer")
    st.caption("Шаг 0 · проверка облачного запуска")

    try:
        repo_id = st.secrets.get("SENTIMENT_REPO_ID", "")
        revision = st.secrets.get("SENTIMENT_REVISION", "main")
    except StreamlitSecretNotFoundError:
        repo_id = ""
    if not repo_id:
        st.info('Укажи SENTIMENT_REPO_ID = "имя/репозиторий" в настройках Secrets.')
        return

    short = "Тапсырыс уақытында келді. Сапасы жақсы, маған ұнады!"
    medium = (
        "Тапсырысым екі күн кешігіп келді. Алғашында қолдау қызметінен жауап ала алмадым. "
        "Кейін қызметкер хабарласып, жағдайды түсіндірді. Тауардың сапасы жақсы екен. "
        "Жеткізу біршама қиын болды, бірақ соңында бәрі шешілді."
    )
    long = ((medium + " ") * (MAX_CHARS // len(medium) + 1))[: MAX_CHARS - 1] + "."
    presets = {"Короткий отзыв": short, "Средний текст": medium, "Текст на 3 000 символов": long}

    example = st.selectbox("Пример для проверки", presets)
    with st.form("analysis"):
        raw_text = st.text_area("Казахский текст", value=presets[example], key=example,
                                max_chars=MAX_CHARS, height=180)
        submitted = st.form_submit_button("Талдау", type="primary")
    if not submitted:
        return

    text = clean_text(raw_text)
    if not text:
        st.warning("Введи текст для анализа.")
        return

    started = perf_counter()
    with st.spinner("Загружаем модели и анализируем текст…"):
        try:
            sentiment_model, tokenizer, encoder = load_models(repo_id.strip(), revision)
            loaded = perf_counter()
            polarity = predict_polarity(sentiment_model, text)
            text_tokens, chunks, max_chunk_tokens = check_encoder(tokenizer, encoder, text)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).exception("Не удалось загрузить модели или обработать текст")
            st.error(f"Проверка остановилась. {exc}")
            return

    finished = perf_counter()
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    left, right = st.columns(2)
    left.metric("Тональдылық · v2 → 3", polarity)
    right.metric("Базовый энкодер E5", "Работает")
    st.caption("Метка эмоции появится после подключения дообученной модели.")
    st.write(f"**Символы** · {len(raw_text)}  **Токены** · {text_tokens}  **Фрагменты** · {chunks}")
    st.write(f"**Максимум в одном фрагменте** · {max_chunk_tokens}/{MAX_TOKENS} токенов")
    st.write(f"**Загрузка** · {loaded - started:.2f} с  **Анализ** · {finished - loaded:.2f} с")
    st.write(f"**Пиковый RSS процесса** · {peak_mib:,.0f} МиБ  **PyTorch** · {torch.__version__}")


if __name__ == "__main__":
    main()
