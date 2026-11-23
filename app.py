"""Kazakh NLP Analyzer: тональность (TF-IDF + LogReg) и эмоции (дообученная E5-small)."""

import base64
import hashlib
import html
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
# P(positive) >= 0.66 -> positive. Полоса «Анықталмады» шириной ±0.10 вокруг порога
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
        raise ValueError(
            f"requirements.txt ішінде scikit-learn=={expected} орнатыңыз "
            f"(қазір {sklearn.__version__} орнатылған)."
        )

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
    return {
        "sentiment": sentiment_model,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "head": head,
        "emo_info": emo_info,
    }


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

    info, tokenizer, encoder, head = (
        models["emo_info"],
        models["tokenizer"],
        models["encoder"],
        models["head"],
    )
    probs = []
    with torch.inference_mode():
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = [info["tokenizer_prefix"] + chunk for chunk in chunks[start : start + BATCH_SIZE]]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=info["max_length"],
                return_tensors="pt",
            )
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


def _tone_badge_class(label):
    return {
        "Оң": "nlp-badge-pos",
        "Теріс": "nlp-badge-neg",
        "Аралас": "nlp-badge-mixed",
        "Бейтарап": "nlp-badge-neutral",
        "Анықталмады": "nlp-badge-neutral",
    }.get(label, "nlp-badge-neutral")


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
    return {
        "chunks": chunks,
        "weights": weights,
        "p_pos": p_pos,
        "emo": emo,
        "max_chunk_tokens": max(weights) + prefix_len + tokenizer.num_special_tokens_to_add(pair=False),
    }


def show_results(result, labels, labels_kk):
    chunks, weights, p_pos, emo = result["chunks"], result["weights"], result["p_pos"], result["emo"]
    overall_pos = float(weighted_mean(p_pos, weights))
    overall_emo = weighted_mean(emo, weights)
    most_neg, most_pos = int(np.argmin(p_pos)), int(np.argmax(p_pos))
    mixed = p_pos[most_neg] <= NEG_THRESHOLD and p_pos[most_pos] >= POS_THRESHOLD
    chunk_emotions = [labels_kk[labels[index]] for index in np.argmax(emo, axis=1)]

    tone_text = "Аралас" if mixed else polarity_label(overall_pos)
    tone_cls = _tone_badge_class(tone_text)
    marker_pct = max(2.0, min(98.0, overall_pos * 100.0))

    if mixed:
        right_body = f"""
            <div class="nlp-card-label">Үзінділердегі контраст эмоциялар</div>
            <div class="nlp-mixed-list">
                <div class="nlp-mixed-item">
                    <span>Теріс үзінді №{most_neg + 1}</span>
                    <strong>{html.escape(chunk_emotions[most_neg])}</strong>
                </div>
                <div class="nlp-mixed-item">
                    <span>Оң үзінді №{most_pos + 1}</span>
                    <strong>{html.escape(chunk_emotions[most_pos])}</strong>
                </div>
            </div>
            <div class="nlp-subnote">Мәтінде айқын оң және айқын теріс үзінділер қатар табылды.</div>
        """
    else:
        top_idx = int(np.argmax(overall_emo))
        top_emo = labels_kk[labels[top_idx]]
        top_score = float(overall_emo[top_idx]) * 100.0
        right_body = f"""
            <div class="nlp-card-label">Эмоцияның басым белгісі (E5)</div>
            <div class="nlp-card-value-row">
                <span class="nlp-card-value">{html.escape(top_emo)}</span>
                <span class="nlp-score">{top_score:.1f}%</span>
            </div>
            <div class="nlp-subnote">
                Үзінділер бойынша токен санымен өлшенген орташа баға.
                Модель ықтималдықтары бөлек калибрленбеген.
            </div>
        """

    st.html(f"""
        <section class="nlp-grid-2" lang="kk">
            <div class="nlp-card">
                <div class="nlp-card-label">Тональдылық (TF-IDF + LogReg)</div>
                <div class="nlp-card-value-row">
                    <span class="nlp-card-value">{html.escape(tone_text)}</span>
                    <span class="nlp-badge {tone_cls}">P(оң) = {overall_pos:.2f}</span>
                </div>
                <div class="nlp-scale-wrap">
                    <div class="nlp-scale-track">
                        <div class="nlp-scale-threshold" title="Шек {POLARITY_THRESHOLD:.2f}"></div>
                        <div class="nlp-scale-marker" style="left: {marker_pct:.1f}%;"></div>
                    </div>
                    <div class="nlp-scale-ticks">
                        <span>0.00 Теріс</span>
                        <span>{NEG_THRESHOLD:.2f}–{POS_THRESHOLD:.2f} шек ({POLARITY_THRESHOLD:.2f})</span>
                        <span>1.00 Оң</span>
                    </div>
                </div>
            </div>
            <div class="nlp-card">
                {right_body}
            </div>
        </section>
    """)

    st.html("""
        <div class="nlp-section-head">
            <h3>Эмоция моделінің бағалары</h3>
            <span>E5-small · 6 санат</span>
        </div>
    """)
    show_emotion_chart(labels, labels_kk, overall_emo)

    if len(chunks) > 1:
        st.html(f"""
            <div class="nlp-section-head">
                <h3>Үзінділер бойынша талдау</h3>
                <span>{len(chunks)} үзінді · {sum(weights)} токен</span>
            </div>
        """)
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
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        neg_title = (
            "Ең теріс үзінді"
            if p_pos[most_neg] <= NEG_THRESHOLD
            else "Ең аз оң үзінді (айқын теріс жоқ)"
        )
        pos_title = (
            "Ең оң үзінді"
            if p_pos[most_pos] >= POS_THRESHOLD
            else "Ең оңға жақын үзінді (айқын оң жоқ)"
        )
        with st.expander("Тональдылығы бойынша шеткі үзінділер"):
            st.html(f"""
                <div class="nlp-excerpt-grid">
                    <div class="nlp-excerpt nlp-excerpt-neg">
                        <div class="nlp-excerpt-meta">
                            <span>{html.escape(neg_title)} · №{most_neg + 1}</span>
                            <code>P(оң) = {p_pos[most_neg]:.2f}</code>
                        </div>
                        <p>{html.escape(chunks[most_neg])}</p>
                    </div>
                    <div class="nlp-excerpt nlp-excerpt-pos">
                        <div class="nlp-excerpt-meta">
                            <span>{html.escape(pos_title)} · №{most_pos + 1}</span>
                            <code>P(оң) = {p_pos[most_pos]:.2f}</code>
                        </div>
                        <p>{html.escape(chunks[most_pos])}</p>
                    </div>
                </div>
            """)

    st.html(f"""
        <div class="nlp-footnote">
            Тональдылық <strong>«Анықталмады»</strong> деп шықса, баға белгісіздік аймағында
            (<span class="nlp-mono">{NEG_THRESHOLD:.2f}–{POS_THRESHOLD:.2f}</span>).
            Негізгі шек <span class="nlp-mono">{POLARITY_THRESHOLD:.2f}</span> валидацияда таңдалған,
            ал <span class="nlp-mono">±{NEUTRAL_HALF_WIDTH:.2f}</span> жолағы эвристикалық ережемен берілген.
            Эмоция моделі шағын жиында үйретілген және 60 мәтіндік пилотта бағаланған.
        </div>
    """)


def show_emotion_chart(labels, labels_kk, scores):
    import altair as alt

    data = pd.DataFrame({
        "emotion": [labels_kk[label] for label in labels],
        "score": scores,
        "leading": np.arange(len(scores)) == int(np.argmax(scores)),
    }).sort_values("score", ascending=False, kind="mergesort")

    order = data["emotion"].tolist()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "emotion:N",
            title=None,
            sort=order,
            axis=alt.Axis(labelLimit=140, labelPadding=10, ticks=False, domain=False),
        ),
        x=alt.X(
            "score:Q",
            scale=alt.Scale(domain=[0, 1.12]),
            axis=alt.Axis(format=".0%", title=None, tickCount=5, ticks=False, domain=False),
        ),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, height=18).encode(
        color=alt.condition("datum.leading", alt.value("#0052FF"), alt.value("#CBD5E1")),
        tooltip=[
            alt.Tooltip("emotion:N", title="Эмоция"),
            alt.Tooltip("score:Q", title="Орташа баға", format=".1%"),
        ],
    )
    labels_layer = base.mark_text(
        align="left", baseline="middle", dx=8, fontSize=12, fontWeight="bold"
    ).encode(
        text=alt.Text("score:Q", format=".1%"),
        color=alt.condition("datum.leading", alt.value("#0052FF"), alt.value("#475569")),
    )
    chart = (
        (bars + labels_layer)
        .properties(height=216)
        .configure(background="transparent", font="Inter, NLP Sans, sans-serif")
        .configure_view(stroke=None)
        .configure_axis(labelFontSize=12, labelColor="#334155", gridColor="#E2E8F0", gridDash=[3, 3])
    )
    st.altair_chart(chart, width="stretch", theme=None)


def show_header():
    assets = Path(__file__).parent / "assets"
    stylesheet = assets / "design.css"
    if stylesheet.is_file():
        css = stylesheet.read_text(encoding="utf-8")
        for placeholder, filename in (
            ("__SANS__", "noto-sans.woff"),
            ("__SANS_BOLD__", "noto-sans-bold.woff"),
        ):
            font = assets / filename
            if font.is_file():
                css = css.replace(placeholder, base64.b64encode(font.read_bytes()).decode("ascii"))
        st.html(f"<style>{css}</style>")

    st.html(f"""
        <header class="nlp-header" lang="kk">
            <div class="nlp-header-main">
                <div class="nlp-kicker">
                    <span class="nlp-kicker-dot"></span>
                    Kazakh NLP · Оқу-зерттеу жобасы
                </div>
                <h1 lang="en">Kazakh NLP Analyzer</h1>
                <p>Қазақ тіліндегі пікірлердің тональдылығы мен эмоциялық реңкін
                   меншікті ML-модельдер және Gemini API арқылы талдау құралы.</p>
            </div>
            <div class="nlp-meta-strip">
                <div class="nlp-meta-pill">KazSAnDRA <strong>F1 0.801</strong></div>
                <div class="nlp-meta-pill">Шек <strong>{POLARITY_THRESHOLD:.2f}</strong></div>
                <div class="nlp-meta-pill">E5 пилот <strong>0.722</strong></div>
                <div class="nlp-meta-pill">Gemini <strong>қаз / орыс</strong></div>
            </div>
        </header>
    """)


def show_project_details():
    st.html(f"""
        <aside class="nlp-project" lang="kk" aria-labelledby="nlp-project-title">
            <div class="nlp-project-head">
                <h2 id="nlp-project-title">Жоба туралы</h2>
                <span class="nlp-project-tag">v3 · 2026</span>
            </div>
            <div class="nlp-pipeline">
                <span class="step">Мәтін</span> →
                <span class="step">Үзінділер</span> →
                <span class="step">TF-IDF + E5</span> /
                <span class="step">Gemini</span>
            </div>
            <dl>
                <div>
                    <dt>
                        <span>Пікірдің тональдылығы</span>
                        <span class="nlp-score">F1 0.8014</span>
                    </dt>
                    <dd>TF-IDF және логистикалық регрессия. KazSAnDRA тестіндегі
                        <span class="nlp-mono">16 797</span> бүтін пікірде шекті
                        <span class="nlp-mono">0.50 → {POLARITY_THRESHOLD:.2f}</span> ауыстыру macro F1 мәнін
                        <span class="nlp-mono">0.7819</span>-дан <span class="nlp-mono">0.8014</span>-ке өсірді
                        (теріс класс recall: <span class="nlp-mono">0.539 → 0.669</span>).</dd>
                </div>
                <div>
                    <dt>
                        <span>Эмоциялық реңк (E5)</span>
                        <span class="nlp-score">43 / 60</span>
                    </dt>
                    <dd>Қосымша үйретілген E5-small алты эмоцияны бағалайды.
                        Қазақша синтетикалық пилотта (<span class="nlp-mono">60</span> мәтін)
                        macro F1 <span class="nlp-mono">0.436 → 0.722</span> өсті
                        (салыстыру: Kaz-RoBERTa — <span class="nlp-mono">45/60</span>, F1 <span class="nlp-mono">0.753</span>).</dd>
                </div>
                <div>
                    <dt>
                        <span>Gemini арқылы талдау</span>
                        <span class="nlp-score">20 000 таңба</span>
                    </dt>
                    <dd>Қазақша және орысша ұзын мәтіндердің тақырыбын, қысқаша мазмұнын,
                        тональдылығы мен эмоцияларын береді. Тек бастапқы мәтінде дәл кездесетін
                        дәйексөздер ғана сақталады.</dd>
                </div>
            </dl>
        </aside>
    """)


def main():
    st.set_page_config(page_title="Kazakh NLP Analyzer", layout="wide")
    show_header()
    review_mode, gemini_mode = "Қазақша пікір (TF-IDF + E5)", "Gemini арқылы талдау"
    with st.container(key="nlp_workspace"):
        workspace, project = st.columns([2.25, 1], gap="large")
        with workspace:
            mode = st.radio(
                "Талдау түрі",
                (review_mode, gemini_mode),
                horizontal=True,
                label_visibility="collapsed",
            )
            if mode == gemini_mode:
                show_gemini_mode()
            else:
                show_review_mode()
        with project:
            show_project_details()


def show_gemini_mode():
    st.caption("Gemini қазақша немесе орысша мәтіннің тақырыбын, тональдылығы мен эмоцияларын анықтап, қысқаша мазмұнын береді.")
    try:
        from gemini_analysis import (
            DEFAULT_MODEL,
            MAX_TEXT_CHARS,
            GeminiAnalysisError,
            analyze_with_gemini,
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
            "Қазақша немесе орысша мәтін",
            max_chars=MAX_TEXT_CHARS,
            height=220,
            placeholder="Талдауға арналған мәтінді осында енгізіңіз…",
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
    tone_kk = tone_labels[result.tone]
    tone_cls = _tone_badge_class(tone_kk)

    st.html(f"""
        <section class="nlp-grid-2" lang="kk">
            <div class="nlp-card">
                <div class="nlp-card-label">Тақырып</div>
                <div class="nlp-card-value" style="font-size: 1.12rem; line-height: 1.38;">
                    {html.escape(result.topic)}
                </div>
            </div>
            <div class="nlp-card">
                <div class="nlp-card-label">Тональдылық (Gemini)</div>
                <div class="nlp-card-value-row">
                    <span class="nlp-card-value">{html.escape(tone_kk)}</span>
                    <span class="nlp-badge {tone_cls}">{html.escape(model)}</span>
                </div>
            </div>
        </section>
        <div class="nlp-card" style="margin-bottom: 1rem;">
            <div class="nlp-card-label">Қысқаша мазмұн</div>
            <p style="margin: 0; font-size: 0.94rem; line-height: 1.62; color: #334155;">
                {html.escape(result.summary)}
            </p>
        </div>
    """)

    st.html("""
        <div class="nlp-section-head">
            <h3>Мәтіндегі эмоциялар мен дәйексөздер</h3>
            <span>Тікелей дәйексөз тексерісі</span>
        </div>
    """)
    if result.emotions:
        cards = []
        for emotion in result.emotions:
            cards.append(f"""
                <div class="nlp-quote-card">
                    <div class="nlp-quote-top">
                        <span class="nlp-quote-emotion">{html.escape(emotion.label)}</span>
                        <span class="nlp-quote-verified">✓ Мәтінде расталды</span>
                    </div>
                    <p class="nlp-quote-text">«{html.escape(emotion.quote)}»</p>
                </div>
            """)
        st.html(f'<div class="nlp-quote-list">{"".join(cards)}</div>')
    else:
        st.info("Мәтінде айқын эмоция анықталмады.")


def _apply_preset(sample_text):
    st.session_state["review_text_input"] = sample_text


def show_review_mode():
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
    short_mixed = "Жеткізу кешікті, бірақ тауар ұнады."
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
    presets = [
        ("Оң пікір", short),
        ("Теріс пікір", negative),
        ("Қысқа аралас", short_mixed),
        ("Аралас пікір", mixed),
        ("Орташа пікір", medium),
        ("Ұзын пікір (үстел)", LONG_REVIEW),
    ]

    if "review_text_input" not in st.session_state:
        st.session_state["review_text_input"] = ""

    st.html('<div class="nlp-preset-label">Жылдам тексеру үлгілері</div>')
    with st.container(key="nlp_presets"):
        for row_start in range(0, len(presets), 3):
            cols = st.columns(3, gap="small")
            for col, (label, sample) in zip(cols, presets[row_start : row_start + 3]):
                col.button(
                    label,
                    key=f"preset_{label}",
                    on_click=_apply_preset,
                    args=(sample,),
                    width="stretch",
                )
        extra_left, extra_right = st.columns([2, 1], gap="small")
        extra_left.button(
            "Ұзын мәтін (коммуналдық төлем)",
            key="preset_long_story",
            on_click=_apply_preset,
            args=(LONG_STORY,),
            width="stretch",
        )
        extra_right.button(
            "Тазалау",
            key="preset_clear",
            on_click=_apply_preset,
            args=("",),
            width="stretch",
        )

    with st.form("analysis"):
        raw_text = st.text_area(
            "Қазақша пікір",
            key="review_text_input",
            max_chars=MAX_CHARS,
            height=190,
            placeholder="Пікір мәтінін осында енгізіңіз немесе жоғарыдағы үлгіні таңдаңыз…",
        )
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
        st.write("Тональдылық моделі — TF-IDF және логистикалық регрессия. Эмоция моделі — E5.")
        st.write(
            f"**Таңбалар** · {len(raw_text)}  **Токендер** · {sum(result['weights'])}  "
            f"**Үзінділер** · {len(result['chunks'])}"
        )
        st.write(f"**Бір үзіндідегі максимум** · {result['max_chunk_tokens']}/{info['max_length']} токен")
        st.write(f"**Жүктеу** · {loaded - started:.2f} сек  **Талдау** · {finished - loaded:.2f} сек")
        st.write(f"**Процестің ең жоғары RSS** · {peak_mib:,.0f} МиБ  **PyTorch** · {version('torch')}")


if __name__ == "__main__":
    main()
