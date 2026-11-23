"""Kazakh NLP Analyzer: тональность (TF-IDF + LogReg) и эмоции (дообученная E5-small)."""

import hashlib
import json
import logging
import re
import resource
import unicodedata
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Тональность: файлы лежат в репозитории из секрета SENTIMENT_REPO_ID.
MODEL_FILE = "tfidf_logreg_polarity_v3.joblib"
MODEL_INFO = "metrics_polarity_v3.json"
# Эмоции: публичный репозиторий, версия зафиксирована коммитом.
EMOTION_REPO = "radiuuum/kazakh_nlp_emotion_e5"
EMOTION_REVISION = "39ab9bbdfd12dcdda4313fdb9f13a07bfdd7f6e1"

MAX_CHARS = 3_000
BATCH_SIZE = 8
# Порог полярности подобран на validation (threshold_polarity_v3.json):
# P(positive) >= 0.66 -> positive. Полоса «Бейтарап» шириной ±0.10 вокруг порога
# это эвристика, а не отдельный обученный класс.
POLARITY_THRESHOLD = 0.66
NEUTRAL_HALF_WIDTH = 0.10
NEG_THRESHOLD = round(POLARITY_THRESHOLD - NEUTRAL_HALF_WIDTH, 2)  # 0.56
POS_THRESHOLD = round(POLARITY_THRESHOLD + NEUTRAL_HALF_WIDTH, 2)  # 0.76

LONG_REVIEW = """Өткен аптада үйге жұмыс істеуге ыңғайлы үстелге тапсырыс бердім. Сайттан өлшемін, түсін және жеткізу уақытын алдын ала таңдадым. Төлемнен кейін растау хаты бірден келді. Суреттері анық, сипаттамасы түсінікті болды. Тапсырысты рәсімдеу кезінде аса көп уақыт жұмсамадым. Бағасы басқа дүкендермен салыстырғанда қолайлы көрінді.

Жеткізу жұма күні түстен кейін болады деп көрсетілген еді. Сол күні таңертең курьер телефон соғып, мекенжайды нақтылады. Алайда келісілген уақытта ешкім келмеді. Хабарлама да, жаңа уақыт туралы ескерту де болған жоқ. Мен бірнеше сағат үйде күтіп, басқа жоспарларымды кейінге қалдырдым. Уақыттың өзгергенін ертерек білсем, күнімді басқаша жоспарлар едім.

Кешке қолдау қызметіне жаздым. Алғашқы жауап тым жалпылама болды. Қызметкер тапсырыс нөмірін сұрап, жағдайды тексеретінін айтты. Біраз уақыттан кейін екінші қызметкер хабарласты. Ол қоймада жүк кешігіп тиелгенін түсіндіріп, келесі күнге жаңа уақыт ұсынды. Осы жолы нақты ақпарат берілгені ұнады. Дегенмен бірінші өтінішке жауап алу үшін көп күтуге тура келді.

Келесі күні курьер уақытында келді. Қорап бүтін екен, ішіндегі бөлшектер жеке оралған. Үстелдің түсі сайттағы суретіне сай болды. Аяқтары орнықты, бұрандалары түгел және бетінде сызат жоқ. Нұсқаулықтағы суреттердің арқасында оны шамамен қырық минутта жинап шықтым. Жиналған үстелге ноутбук, кітаптар және шағын шам еркін сыйды.

Бірнеше күн қолданғаннан кейін үстелдің сапасына көңілім толды. Жұмыс істеген кезде үстел шайқалмайды. Беті күңгірт болғандықтан, жарық шағылыспайды. Кешке оны дымқыл шүберекпен сүртіп шықтым, із қалған жоқ. Маған әсіресе бұрыштарының жұмырланғаны ұнады. Үйде орын аз болса да, үстел бөлмеге жақсы үйлесті.

Тапсырыстың жағымды жағы тауардың сапасы мен жинаудың оңайлығы болды. Жеткізудің кешігуіне байланысты мазасыздандым, кейін жағдай толық шешілгенде көңілім орнына түсті. Қолдау қызметінің екінші маманы сыпайы сөйлесіп, нақты жауап берді. Егер дүкен жеткізу уақыты өзгергенде бірден хабарлама жіберсе, қызмет сапасы айтарлықтай жақсарар еді."""

LONG_STORY = """Қала шетіндегі ескі бес қабатты үйге көшіп келгенде, ең басты қиындық сықырлаған еден мен әр қабатқа тоқтап, үйдің ішін тексергендей баяу жүретін лифт болады деп ойлағанмын. Бірақ нағыз ересек өмірмен танысу қараша айында, пошта жәшігіне күздің алғашқы коммуналдық төлем түбіртегі түскен кезде басталды.

«Барлығы» деген бағандағы соманы көргенде, қағазды үш рет қайта оқып шықтым. Көрсетілген ақшаға қарасаң, түні бойы шағын екі бөлмелі пәтерді емес, бассейні бар үлкен тропикалық жылыжайды жылытып отырғандай көрінеді. Бірінші ойыма келгені — бәрімізге таныс ескі жол: қағазды басып шығарып, жеке куәлікті алып, таңертеңнен бастап кез келген заңгерден артық сөйлейтін зейнеткерлер жиналатын жергілікті есеп айырысу орталығына бару.

Сыртқа шығар алдында баспалдақ алаңында көршім Валентина апаймен кездесіп қалдым. Қолымдағы түбіртек пен абдыраған түрімді көрген ол бәрін бірден түсініп, жұма күні таңертең кеңсеге бару тек жүйкесі мықты адамдардың ісі екенін айтып күлді. Оның сөзінше, жақында үйде есепке алу жүйесі ауысқандықтан, кіреберістің жартысына деректер қате беріліпті, ал мұндай мәселені қапырық дәлізде кезек күтіп емес, онлайн шешкен әлдеқайда оңай екен.

Асүйге қайта кіріп, қою шай демдеп алдым да, ноутбукты ашып zhkh.com сайтына кірдім. Шынымды айтсам, ешқандай үміт болған жоқ: коммуналдық мәселелер апталап созылып, ондаған анықтама жинауды талап етеді деген ойға үйреніп қалғанбыз ғой. Бірақ бәрі әлдеқайда қарапайым болып шықты. Өткен айлардың есептеу мұрағатын ашып, жеке есептегіштердің көрсеткіштерін жалпы үйлік есептермен салыстырып, шатасудың себебін тез таптым. Сөйтсем, пәтердің бұрынғы иесі есептегіш құралдардың бірін тексеру мерзімін өткізіп алыпты, ал жүйе мен жалғыз тұрсам да, шығынды үш адамға арналған жоғары нормативпен есептей бастаған екен.

Одан әрі бәрі тез шешілді. Сервис арқылы өтініш рәсімдеп, келесі күні-ақ жасатып алған жаңа тексеру актісін жүктедім де, қазіргі көрсеткіштердің фотосуреттерін тіркедім. Ешқандай кезек те, жоғалып кететін өтініштер де, «Түскі үзіліс 13:00-ден 14:00-ге дейін» деген жазуы бар терезе алдындағы дау-дамай да болған жоқ.

Келесі есептік кезеңде-ақ түбіртекте ұқыпты қайта есептеу пайда болып, артық төленген сома алдағы төлемдердің есебіне жазылды. Мұны естіген Валентина апай басын изеп, ендігіде көрсеткіштерді қағазға жазып, кіреберістегі жәшікке тастап жүрмес үшін өзіне де жеке кабинет ашып беруді сұрады. Осылайша қарапайым тұрмыстық жағдай көршілермен жақынырақ араласуға себеп болып, тіпті коммуналдық мәселелерді де артық жүйке жұқартпай шешуге болатынын көрсетті."""


def clean_text(text):
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@st.cache_resource(show_spinner=False)
def load_models(repo_id, revision):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

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
    import torch

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
    return "Анықталмады"


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
    most_neg, most_pos = int(np.argmin(p_pos)), int(np.argmax(p_pos))
    mixed = p_pos[most_neg] <= NEG_THRESHOLD and p_pos[most_pos] >= POS_THRESHOLD
    chunk_emotions = [labels_kk[labels[index]] for index in np.argmax(emo, axis=1)]

    left, right = st.columns(2)
    left.metric("Тональдылық", "Аралас" if mixed else polarity_label(overall_pos))
    left.caption(f"Орташа P(оң) = {overall_pos:.2f}")
    if mixed:
        right.write("**Үзінділердегі эмоция белгілері**")
        right.write(f"Теріс үзінді №{most_neg + 1} · {chunk_emotions[most_neg]}")
        right.write(f"Оң үзінді №{most_pos + 1} · {chunk_emotions[most_pos]}")
    else:
        right.metric("Эмоцияның басым белгісі", labels_kk[labels[int(np.argmax(overall_emo))]])

    st.write("**Бүкіл мәтін бойынша эмоция моделінің бағалары**")
    st.bar_chart(pd.DataFrame({"Орташа баға": overall_emo}, index=[labels_kk[l] for l in labels]))

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
                "Эмоция": chunk_emotions[i],
                "Модель бағасы": round(float(emo[i][top]), 2),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        neg_title = ("Ең теріс" if p_pos[most_neg] <= NEG_THRESHOLD
                     else "Ең аз оң (айқын теріс үзінділер жоқ)")
        pos_title = ("Ең оң" if p_pos[most_pos] >= POS_THRESHOLD
                     else "Ең оңға жақын (айқын оң үзінділер жоқ)")
        with st.expander("Тональдылығы бойынша шеткі үзінділер"):
            st.markdown(f"**{neg_title} (№{most_neg + 1}, P(оң) = {p_pos[most_neg]:.2f}):** {chunks[most_neg]}")
            st.markdown(f"**{pos_title} (№{most_pos + 1}, P(оң) = {p_pos[most_pos]:.2f}):** {chunks[most_pos]}")

    st.caption("Тональдылық «Анықталмады» деп көрсетілсе, модель бағасы белгісіздік аймағында. "
               f"Негізгі шек {POLARITY_THRESHOLD} валидацияда таңдалған, "
               f"ал ±{NEUTRAL_HALF_WIDTH} аймағы эвристикалық ережемен берілген. "
               "Эмоция моделі шағын жиында үйретілген және 60 мәтіндік пилотта бағаланған.")


def main():
    st.set_page_config(page_title="Kazakh NLP Analyzer", layout="centered")
    st.title("Kazakh NLP Analyzer")
    review_mode, gemini_mode = "Қазақша пікірді талдау", "Мәтінді кеңейтілген талдау"
    mode = st.radio("Талдау түрі", (review_mode, gemini_mode), horizontal=True)
    if mode == gemini_mode:
        show_gemini_mode()
    else:
        show_review_mode()


def show_gemini_mode():
    try:
        from gemini_analysis import (
            DEFAULT_MODEL, MAX_TEXT_CHARS, GeminiAnalysisError, analyze_with_gemini,
        )
    except ImportError:
        st.error("Gemini режимі үшін google-genai пакетін орнатыңыз.")
        return

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        model = st.secrets.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    except StreamlitSecretNotFoundError:
        api_key, model = "", DEFAULT_MODEL

    if not api_key:
        st.info("Gemini режимін қосу үшін Secrets бөлімінде GEMINI_API_KEY көрсетіңіз.")
        return

    with st.form("gemini_analysis"):
        raw_text = st.text_area(
            "Қазақша немесе орысша мәтін", max_chars=MAX_TEXT_CHARS, height=220,
        )
        st.caption("Талдау батырмасын басқанда мәтін Google Gemini API қызметіне жіберіледі.")
        submitted = st.form_submit_button("Gemini арқылы талдау", type="primary")
    if not submitted:
        return

    text = clean_text(raw_text)
    if not text:
        st.warning("Талдау үшін мәтін енгізіңіз.")
        return

    with st.spinner("Gemini мәтінді талдап жатыр…"):
        try:
            result = analyze_with_gemini(text, api_key, model)
        except (GeminiAnalysisError, ValueError) as exc:
            st.error(str(exc))
            return

    tone_labels = {"positive": "Оң", "negative": "Теріс", "neutral": "Бейтарап", "mixed": "Аралас"}
    st.subheader("Тақырып")
    st.text(result.topic)
    st.subheader("Қысқаша мазмұн")
    st.text(result.summary)
    st.metric("Тональдылық", tone_labels[result.tone])
    st.subheader("Мәтіндегі эмоциялар")
    if result.emotions:
        for emotion in result.emotions:
            st.text(f"{emotion.label} — «{emotion.quote}»")
    else:
        st.write("Мәтінде айқын эмоция анықталмады.")
    st.caption(f"Модель · {model}")


def show_review_mode():
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
    presets = {
        "Қысқа оң пікір": short,
        "Орташа пікір": medium,
        "Теріс пікір": negative,
        "Аралас пікір": mixed,
        "Ұзын пікір (үстел)": LONG_REVIEW,
        "Ұзын мәтін (коммуналдық төлем)": LONG_STORY,
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
        st.write(f"**Процестің ең жоғары RSS** · {peak_mib:,.0f} МиБ  **PyTorch** · {version('torch')}")


if __name__ == "__main__":
    main()
