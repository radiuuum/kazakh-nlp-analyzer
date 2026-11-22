"""Kazakh NLP Analyzer: тональность (TF-IDF + LogReg) и эмоции (дообученная E5-small)."""

import hashlib
import json
import logging
import re
import resource
import unicodedata
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from streamlit.errors import StreamlitSecretNotFoundError

# Тональность: файлы лежат в репозитории из секрета SENTIMENT_REPO_ID.
MODEL_FILE = "tfidf_logreg_polarity_v3.joblib"
MODEL_INFO = "metrics_polarity_v3.json"
# Эмоции: публичный репозиторий, версия зафиксирована коммитом.
EMOTION_REPO = "radiuuum/kazakh_nlp_emotion_e5"
EMOTION_REVISION = "39ab9bbdfd12dcdda4313fdb9f13a07bfdd7f6e1"

MAX_CHARS = 3_000
BATCH_SIZE = 8
# Пороги для метки «Бейтарап». Это эвристика, а не отдельный обученный класс.
POS_THRESHOLD = 0.65
NEG_THRESHOLD = 0.35


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
        raise ValueError(f"SHA-256 файла модели не совпал с {MODEL_INFO}.")
    sentiment_model = joblib.load(model_path)
    if 1 not in list(sentiment_model.classes_):
        raise ValueError("В модели тональности нет класса 1 (positive).")

    emo_info_path = Path(hf_hub_download(EMOTION_REPO, "emotion_model_info.json", revision=EMOTION_REVISION))
    emo_info = json.loads(emo_info_path.read_text(encoding="utf-8"))
    head = load_file(hf_hub_download(EMOTION_REPO, emo_info["head_file"], revision=EMOTION_REVISION))
    if head["weight"].shape[0] != len(emo_info["label_order"]):
        raise ValueError("Число выходов головы эмоций не совпало с label_order.")

    torch.set_num_threads(2)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(EMOTION_REPO, revision=EMOTION_REVISION)
    encoder = BertModel.from_pretrained(EMOTION_REPO, revision=EMOTION_REVISION, use_safetensors=True).eval()
    return {"sentiment": sentiment_model, "tokenizer": tokenizer, "encoder": encoder,
            "head": head, "emo_info": emo_info}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def token_lengths(tokenizer, pieces):
    return [len(ids) for ids in tokenizer(pieces, add_special_tokens=False)["input_ids"]]


def make_chunks(tokenizer, text, budget):
    """Режет текст на фрагменты по границам предложений, не длиннее budget токенов."""
    sentences = split_sentences(text)
    lengths = token_lengths(tokenizer, sentences)
    chunks, current, current_len = [], [], 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
            current, current_len = [], 0

    for sentence, length in zip(sentences, lengths):
        if length > budget:  # слишком длинное предложение режем по словам
            flush()
            words = sentence.split()
            for word, word_len in zip(words, token_lengths(tokenizer, words)):
                if current_len + word_len > budget:
                    flush()
                current.append(word)
                current_len += word_len
            flush()
        else:
            if current_len + length > budget:
                flush()
            current.append(sentence)
            current_len += length
    flush()
    return chunks


def predict_polarity(model, chunks):
    """Вероятность класса positive для каждого фрагмента."""
    index = list(model.classes_).index(1)
    return model.predict_proba(chunks)[:, index]


def predict_emotions(models, chunks):
    info, tokenizer, encoder, head = (models["emo_info"], models["tokenizer"],
                                      models["encoder"], models["head"])
    probs = []
    with torch.inference_mode():
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = [info["tokenizer_prefix"] + chunk for chunk in chunks[start : start + BATCH_SIZE]]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=info["max_length"], return_tensors="pt")
            hidden = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            logits = pooled @ head["weight"].T + head["bias"]
            if not torch.isfinite(logits).all().item():
                raise ValueError("Модель эмоций вернула некорректные значения.")
            probs.append(torch.softmax(logits, dim=-1))
    return torch.cat(probs).numpy()


def weighted_mean(values, weights):
    return np.average(np.asarray(values), axis=0, weights=np.asarray(weights, dtype=float))


def polarity_label(p_pos):
    if p_pos >= POS_THRESHOLD:
        return "Оң"
    if p_pos <= NEG_THRESHOLD:
        return "Теріс"
    return "Бейтарап"


def analyze(models, text):
    tokenizer, info = models["tokenizer"], models["emo_info"]
    prefix_len = token_lengths(tokenizer, [info["tokenizer_prefix"].strip()])[0]
    budget = info["max_length"] - tokenizer.num_special_tokens_to_add(pair=False) - prefix_len - 2
    chunks = make_chunks(tokenizer, text, budget)
    if not chunks:
        raise ValueError("После обработки не осталось текста для анализа.")
    weights = token_lengths(tokenizer, chunks)
    p_pos = predict_polarity(models["sentiment"], chunks)
    emo = predict_emotions(models, chunks)
    return {"chunks": chunks, "weights": weights, "p_pos": p_pos, "emo": emo,
            "max_chunk_tokens": max(weights) + prefix_len + tokenizer.num_special_tokens_to_add(pair=False)}


def show_results(result, labels, labels_kk):
    chunks, weights, p_pos, emo = result["chunks"], result["weights"], result["p_pos"], result["emo"]
    overall_pos = float(weighted_mean(p_pos, weights))
    overall_emo = weighted_mean(emo, weights)
    order = np.argsort(-overall_emo)

    left, right = st.columns(2)
    left.metric("Тональдылық", polarity_label(overall_pos))
    left.caption(f"P(оң) = {overall_pos:.2f}")
    right.metric("Эмоция", labels_kk[labels[order[0]]])
    right.caption(f"2-ші: {labels_kk[labels[order[1]]]} ({overall_emo[order[1]]:.0%})")

    if len(chunks) > 1 and (p_pos <= NEG_THRESHOLD).any() and (p_pos >= POS_THRESHOLD).any():
        st.info("В тексте есть и явно положительные, и явно отрицательные фрагменты: отзыв смешанный.")

    st.write("**Распределение эмоций по всему тексту**")
    st.bar_chart(pd.DataFrame({"Ықтималдық": overall_emo}, index=[labels_kk[l] for l in labels]))

    if len(chunks) > 1:
        st.write(f"**Разбор по фрагментам** · {len(chunks)}")
        rows = []
        for i, chunk in enumerate(chunks):
            top = int(np.argmax(emo[i]))
            rows.append({
                "№": i + 1,
                "Фрагмент": chunk[:90] + ("…" if len(chunk) > 90 else ""),
                "Тональдылық": polarity_label(float(p_pos[i])),
                "P(оң)": round(float(p_pos[i]), 2),
                "Эмоция": labels_kk[labels[top]],
                "Сенімділік": round(float(emo[i][top]), 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        most_neg, most_pos = int(np.argmin(p_pos)), int(np.argmax(p_pos))
        with st.expander("Самый отрицательный и самый положительный фрагменты"):
            st.markdown(f"**Самый отрицательный (№{most_neg + 1}, P(оң) = {p_pos[most_neg]:.2f}):** {chunks[most_neg]}")
            st.markdown(f"**Самый положительный (№{most_pos + 1}, P(оң) = {p_pos[most_pos]:.2f}):** {chunks[most_pos]}")

    st.caption("«Бейтарап» в тональности означает, что модель не уверена (пороги "
               f"{NEG_THRESHOLD}–{POS_THRESHOLD}, эвристика), это не отдельный обученный класс. "
               "Модель эмоций дообучена на небольшом наборе и оценена на пилоте из 60 текстов.")


def main():
    st.set_page_config(page_title="Kazakh NLP Analyzer", layout="centered")
    st.title("Kazakh NLP Analyzer")
    st.caption("Тональность и эмоции казахских отзывов")

    revision = "main"
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
    negative = (
        "Бағасын толық төлеген соң кемінде тауардың бүтін келуін және дүкеннің уәдесінде тұруын күттім. "
        "Оның орнына бірнеше күн бойы бір сұрақты қайталап жазуға тура келді. "
        "Мәселені кім шешетіні де түсініксіз қалды. Әр сөйлескен сайын тапсырыс нөмірін басынан айтып, "
        "суреттерді қайта жібердім. Осынша уақыт жоғалтқаным өкінішті. "
        "Қысқасы, сатып алғаныма да, осы дүкенге сенгеніме де қатты өкінемін. Мүлде ұсынбаймын."
    )
    mixed = (
        "Қорабы мыжылып келді, ішінде бір бөлшек жетіспеді. Қолдау қызметі сол күні хабарласып, "
        "ертеңіне жетіспеген бөлшекті жіберді. Қазір үстелді пайдаланып жүрмін."
    )
    base = " ".join([medium, negative, mixed])
    long = ((base + " ") * (MAX_CHARS // len(base) + 1))[: MAX_CHARS - 1] + "."
    presets = {
        "Короткий позитивный": short,
        "Средний отзыв": medium,
        "Негативный отзыв": negative,
        "Смешанный отзыв": mixed,
        "Длинный текст (3 000 символов)": long,
    }

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
            models = load_models(repo_id.strip(), revision)
            loaded = perf_counter()
            result = analyze(models, text)
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).exception("Не удалось загрузить модели или обработать текст")
            st.error(f"Анализ остановился. {exc}")
            return
    finished = perf_counter()

    info = models["emo_info"]
    show_results(result, info["label_order"], info["label_names_kk"])

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with st.expander("Техническая информация"):
        st.write(f"**Символы** · {len(raw_text)}  **Токены** · {sum(result['weights'])}  "
                 f"**Фрагменты** · {len(result['chunks'])}")
        st.write(f"**Максимум в одном фрагменте** · {result['max_chunk_tokens']}/{info['max_length']} токенов")
        st.write(f"**Загрузка** · {loaded - started:.2f} с  **Анализ** · {finished - loaded:.2f} с")
        st.write(f"**Пиковый RSS процесса** · {peak_mib:,.0f} МиБ  **PyTorch** · {torch.__version__}")


if __name__ == "__main__":
    main()
