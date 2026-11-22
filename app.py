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
        raise ValueError(f"requirements.txt ішінде scikit-learn=={expected} орнатыңыз "
                         f"(қазір {sklearn.__version__} орнатылған).")

    model_path = Path(hf_hub_download(repo_id, MODEL_FILE, revision=revision))
    with model_path.open("rb") as model_file:
        actual_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    if actual_sha256 != info["model_sha256"]:
        raise ValueError(f"Модель файлының SHA-256 мәні {MODEL_INFO} файлымен сәйкес келмеді.")
    sentiment_model = joblib.load(model_path)
    if 1 not in list(sentiment_model.classes_):
        raise ValueError("Тональдылық моделінде 1-класс (positive) жоқ.")

    emo_info_path = Path(hf_hub_download(EMOTION_REPO, "emotion_model_info.json", revision=EMOTION_REVISION))
    emo_info = json.loads(emo_info_path.read_text(encoding="utf-8"))
    head = load_file(hf_hub_download(EMOTION_REPO, emo_info["head_file"], revision=EMOTION_REVISION))
    if head["weight"].shape[0] != len(emo_info["label_order"]):
        raise ValueError("Эмоция басының шығыс саны label_order-мен сәйкес келмеді.")

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
                raise ValueError("Эмоция моделі дұрыс емес мәндер қайтарды.")
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
        raise ValueError("Өңдеуден кейін талдауға мәтін қалмады.")
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
        st.info("Мәтінде айқын оң да, айқын теріс те үзінділер бар: пікір аралас.")

    st.write("**Бүкіл мәтін бойынша эмоциялардың таралуы**")
    st.bar_chart(pd.DataFrame({"Ықтималдық": overall_emo}, index=[labels_kk[l] for l in labels]))

    if len(chunks) > 1:
        st.write(f"**Үзінділер бойынша талдау** · {len(chunks)}")
        rows = []
        for i, chunk in enumerate(chunks):
            top = int(np.argmax(emo[i]))
            rows.append({
                "№": i + 1,
                "Үзінді": chunk[:90] + ("…" if len(chunk) > 90 else ""),
                "Тональдылық": polarity_label(float(p_pos[i])),
                "P(оң)": round(float(p_pos[i]), 2),
                "Эмоция": labels_kk[labels[top]],
                "Сенімділік": round(float(emo[i][top]), 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        most_neg, most_pos = int(np.argmin(p_pos)), int(np.argmax(p_pos))
        neg_title = ("Ең теріс" if p_pos[most_neg] <= NEG_THRESHOLD
                     else "Ең аз оң (айқын теріс үзінділер жоқ)")
        pos_title = ("Ең оң" if p_pos[most_pos] >= POS_THRESHOLD
                     else "Ең оңға жақын (айқын оң үзінділер жоқ)")
        with st.expander("Тональдылығы бойынша шеткі үзінділер"):
            st.markdown(f"**{neg_title} (№{most_neg + 1}, P(оң) = {p_pos[most_neg]:.2f}):** {chunks[most_neg]}")
            st.markdown(f"**{pos_title} (№{most_pos + 1}, P(оң) = {p_pos[most_pos]:.2f}):** {chunks[most_pos]}")

    st.caption("Тональдылықтағы «Бейтарап» белгісі модельдің сенімсіз екенін білдіреді "
               f"(шектері {NEG_THRESHOLD}–{POS_THRESHOLD}, эвристика), бұл бөлек үйретілген класс емес. "
               "Эмоция моделі шағын жиында үйретілген және 60 мәтіндік пилотта бағаланған.")


def main():
    st.set_page_config(page_title="Kazakh NLP Analyzer", layout="centered")
    st.title("Kazakh NLP Analyzer")
    st.caption("Қазақ тіліндегі пікірлердің тональдылығы мен эмоциясын талдау")

    revision = "main"
    try:
        repo_id = st.secrets.get("SENTIMENT_REPO_ID", "")
        revision = st.secrets.get("SENTIMENT_REVISION", "main")
    except StreamlitSecretNotFoundError:
        repo_id = ""
    if not repo_id:
        st.info('Secrets баптауларында SENTIMENT_REPO_ID = "аты/репозиторий" көрсетіңіз.')
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
        "Қысқа оң пікір": short,
        "Орташа пікір": medium,
        "Теріс пікір": negative,
        "Аралас пікір": mixed,
        "Ұзын мәтін (3 000 таңба)": long,
    }

    example = st.selectbox("Тексеруге арналған мысал", presets)
    with st.form("analysis"):
        raw_text = st.text_area("Қазақша мәтін", value=presets[example], key=example,
                                max_chars=MAX_CHARS, height=180)
        submitted = st.form_submit_button("Талдау", type="primary")
    if not submitted:
        return

    text = clean_text(raw_text)
    if not text:
        st.warning("Талдау үшін мәтін енгізіңіз.")
        return

    started = perf_counter()
    with st.spinner("Модельдер жүктелуде, мәтін талданып жатыр…"):
        try:
            models = load_models(repo_id.strip(), revision)
            loaded = perf_counter()
            result = analyze(models, text)
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).exception("Не удалось загрузить модели или обработать текст")
            st.error(f"Талдау тоқтап қалды. {exc}")
            return
    finished = perf_counter()

    info = models["emo_info"]
    show_results(result, info["label_order"], info["label_names_kk"])

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with st.expander("Техникалық ақпарат"):
        st.write(f"**Таңбалар** · {len(raw_text)}  **Токендер** · {sum(result['weights'])}  "
                 f"**Үзінділер** · {len(result['chunks'])}")
        st.write(f"**Бір үзіндідегі максимум** · {result['max_chunk_tokens']}/{info['max_length']} токен")
        st.write(f"**Жүктеу** · {loaded - started:.2f} сек  **Талдау** · {finished - loaded:.2f} сек")
        st.write(f"**Процестің ең жоғары RSS** · {peak_mib:,.0f} МиБ  **PyTorch** · {torch.__version__}")


if __name__ == "__main__":
    main()
