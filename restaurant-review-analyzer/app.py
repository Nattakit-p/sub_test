import re
from collections import Counter

import pandas as pd
import streamlit as st
from pythainlp.util import normalize
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="Restaurant Review Analyzer",
    page_icon="🍽️",
    layout="wide",
)

BRANDS = [
    "MK", "KFC", "Starbucks", "After You", "S&P", "McDonald's",
    "Burger King", "Pizza Hut", "Chester's", "Swensen's",
    "Fuji", "Shabushi", "Bonchon", "Bar-B-Q Plaza", "Gyu-Kaku",
]

LOCATIONS = [
    "เซ็นทรัลลาดพร้าว", "Central Ladprao", "เซ็นทรัลเวิลด์", "Central World",
    "สยาม", "Siam", "กรุงเทพฯ", "กรุงเทพ", "Bangkok", "เชียงใหม่", "Chiang Mai",
    "พัทยา", "Pattaya", "ภูเก็ต", "Phuket", "หาดใหญ่", "Hat Yai",
    "สาขาเซ็นทรัลลาดพร้าว", "สาขาเซ็นทรัลเวิลด์",
]

MENU_ITEMS = [
    "สุกี้หมู", "สุกี้", "ข้าวมันไก่", "ชาเขียว", "ชาไทย", "ผัดไทย",
    "ต้มยำ", "กะเพรา", "ก๋วยเตี๋ยว", "หมูกระทะ", "ข้าวผัด",
    "พิซซ่า", "Pizza", "Burger", "Fried Chicken", "Chicken",
    "Steak", "Pasta", "Coffee", "กาแฟ", "เค้ก", "Cake",
    "น้ำซุป", "ไก่ทอด", "ข้าว", "อาหารทะเล", "Seafood",
]

POSITIVE_WORDS = [
    "อร่อยมาก", "อร่อย", "ดีมาก", "ดี", "เยี่ยม", "ชอบ", "คุ้ม",
    "สด", "หอม", "เข้มข้น", "บริการดี", "ประทับใจ", "สะอาด",
    "รวดเร็ว", "เป็นมิตร", "good", "great", "delicious", "excellent",
    "fresh", "amazing", "friendly", "worth", "crispy", "tasty",
    "reasonable", "nice", "perfect", "love",
]

NEGATIVE_WORDS = [
    "ไม่อร่อย", "ไม่ดี", "แย่", "แพง", "เค็มเกินไป", "หวานเกินไป",
    "รอนาน", "ช้า", "ไม่สด", "บริการแย่", "ไม่คุ้ม", "สกปรก",
    "เย็นชืด", "แออัด", "ช้ามาก", "bad", "terrible", "expensive",
    "slow", "salty", "bland", "cold", "disappointing", "crowded",
    "overpriced", "poor", "worst",
]

TOPIC_KEYWORDS = {
    "อาหาร": [
        "อาหาร", "เมนู", "สุกี้", "ข้าว", "ไก่", "หมู", "ปลา", "pizza",
        "burger", "chicken", "food", "dish", "meal", "fried chicken",
    ],
    "รสชาติ": [
        "อร่อย", "รสชาติ", "หวาน", "เค็ม", "เผ็ด", "เปรี้ยว", "จืด",
        "เข้มข้น", "หอม", "จืดชืด", "delicious", "tasty", "salty",
        "sweet", "spicy", "bland", "flavor", "flavour", "crispy",
    ],
    "ราคา": [
        "ราคา", "แพง", "ถูก", "คุ้ม", "ไม่คุ้ม", "บาท", "price",
        "expensive", "cheap", "worth", "overpriced", "reasonable", "cost",
    ],
    "การบริการ": [
        "บริการ", "พนักงาน", "รอ", "รอนาน", "ช้า", "บริการดี", "บริการแย่",
        "staff", "service", "wait", "slow", "friendly", "fast",
    ],
    "สถานที่": [
        "สาขา", "เซ็นทรัล", "สยาม", "กรุงเทพ", "เชียงใหม่", "พัทยา",
        "ภูเก็ต", "ร้าน", "central", "location", "branch",
    ],
    "ความสะอาด": [
        "สะอาด", "สกปรก", "ห้องน้ำ", "clean", "dirty", "hygiene",
        "restroom", "toilet",
    ],
    "บรรยากาศ": [
        "บรรยากาศ", "เงียบ", "เสียงดัง", "อบอุ่น", "สวย", "นั่งสบาย",
        "crowded", "atmosphere", "quiet", "noisy", "cozy", "comfortable",
    ],
    "การจัดส่ง": [
        "ส่ง", "จัดส่ง", "เดลิเวอรี่", "delivery", "deliver", "ส่งช้า",
        "ส่งเร็ว", "grab", "lineman",
    ],
}

STOPWORDS_EN = {
    "the", "a", "an", "is", "was", "were", "am", "are", "and", "or",
    "to", "of", "in", "on", "at", "for", "with", "this", "that",
    "it", "very", "quite", "but", "my", "i", "we", "they", "he", "she",
    "you", "our", "their", "from", "as", "be", "been", "have", "has",
    "had", "so", "too", "really",
}


# =========================================================
# Text preprocessing
# =========================================================

def extract_metadata(text: str) -> dict:
    """Extract information that should not remain in cleaned text."""
    urls = re.findall(r"https?://[^\s]+|www\.[^\s]+", text, flags=re.I)
    emails = re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    phones = re.findall(r"(?<!\d)(?:0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}|0\d{9})(?!\d)", text)

    return {
        "urls": urls,
        "emails": emails,
        "phones": phones,
    }


def normalize_repeated_chars(text: str) -> str:
    """Reduce repeated characters used for emphasis, e.g. มากกกก -> มาก."""
    return re.sub(r"([A-Za-zก-๙])\1{2,}", r"\1", text)


def clean_text(text: str) -> str:
    """Clean Thai/English review text while keeping useful words."""
    text = re.sub(r"https?://[^\s]+|www\.[^\s]+", " ", text, flags=re.I)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(
        r"(?<!\d)(?:0\d{1,2}[-\s]?\d{3}[-\s]?\d{4}|0\d{9})(?!\d)",
        " ",
        text,
    )

    # PyThaiNLP normalization helps normalize Thai Unicode sequences.
    text = normalize(text)
    text = normalize_repeated_chars(text)

    # Remove emoji/symbols but keep Thai, English, numbers and basic punctuation.
    text = re.sub(r"[^0-9A-Za-zก-๙\s.,!?%:/&'()-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_language(text: str) -> str:
    """Simple language detection based on Thai/English character counts."""
    thai_count = len(re.findall(r"[ก-๙]", text))
    english_count = len(re.findall(r"[A-Za-z]", text))

    if thai_count == 0 and english_count == 0:
        return "Unknown"
    if thai_count >= english_count:
        return "Thai"
    return "English"


def tokenize_text(text: str, language: str) -> list[str]:
    """Tokenize Thai with PyThaiNLP and English with a lightweight tokenizer."""
    if not text:
        return []

    if language == "Thai":
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    elif language == "English":
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text.lower())
        tokens = [t for t in tokens if t not in STOPWORDS_EN]
    else:
        # Mixed language: use PyThaiNLP, then split English punctuation.
        raw = word_tokenize(text, engine="newmm", keep_whitespace=False)
        tokens = []
        for token in raw:
            if re.fullmatch(r"[A-Za-z]+", token):
                token = token.lower()
                if token not in STOPWORDS_EN:
                    tokens.append(token)
            elif re.search(r"[ก-๙]", token) or re.search(r"\d", token):
                tokens.append(token)

    return [token for token in tokens if token.strip()]


# =========================================================
# POS & NER
# =========================================================

def pos_tag_tokens(tokens: list[str], language: str) -> list[tuple[str, str]]:
    """
    POS tagging.
    Thai uses PyThaiNLP POS tagging.
    English uses a small rule-based fallback to keep the app lightweight.
    """
    if not tokens:
        return []

    if language in {"Thai", "Unknown"}:
        try:
            return pos_tag(tokens, corpus="orchid")
        except Exception:
            return [(token, "UNKNOWN") for token in tokens]

    # Lightweight English POS heuristic.
    tags = []
    for token in tokens:
        lower = token.lower()
        if lower.endswith(("ly",)):
            tag = "ADV"
        elif lower.endswith(("ing", "ed")):
            tag = "VERB"
        elif lower in {"good", "great", "delicious", "excellent", "fresh",
                       "amazing", "friendly", "crispy", "tasty", "bad",
                       "terrible", "expensive", "slow", "salty", "bland"}:
            tag = "ADJ"
        elif token[:1].isupper():
            tag = "PROPN"
        else:
            tag = "WORD"
        tags.append((token, tag))
    return tags


def find_dictionary_entities(text: str, candidates: list[str]) -> list[str]:
    """Find dictionary-based entities without duplicates."""
    found = []
    lower_text = text.lower()

    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate.lower() in lower_text and candidate not in found:
            found.append(candidate)

    return found


def extract_entities(text: str) -> dict:
    """
    Hybrid NER:
    1) Dictionary matching for restaurant/brand, location and menu.
    2) Regex for common location patterns such as 'สาขา...'.
    """
    brands = find_dictionary_entities(text, BRANDS)
    locations = find_dictionary_entities(text, LOCATIONS)
    menus = find_dictionary_entities(text, MENU_ITEMS)

    branch_matches = re.findall(
        r"(?:สาขา|branch)\s*([A-Za-zก-๙0-9][A-Za-zก-๙0-9\s-]{1,40})",
        text,
        flags=re.I,
    )
    for branch in branch_matches:
        branch = re.split(r"[,.!?]", branch)[0].strip()
        if branch and branch not in locations:
            locations.append(branch)

    # Extra heuristic: "ร้าน X" where X is a short proper-looking phrase.
    restaurant_matches = re.findall(
        r"ร้าน\s+([A-Za-zก-๙][A-Za-zก-๙0-9'&.-]{1,30})",
        text,
        flags=re.I,
    )
    for name in restaurant_matches:
        if name not in brands and name not in menus:
            brands.append(name)

    return {
        "BRAND / RESTAURANT": brands,
        "LOCATION": locations,
        "MENU / FOOD": menus,
    }


# =========================================================
# Topic & sentiment
# =========================================================

def find_phrases(text: str, phrases: list[str]) -> list[str]:
    """Find matching phrases, avoiding duplicate outputs."""
    found = []
    lower_text = text.lower()

    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase.lower() in lower_text and phrase.lower() not in [x.lower() for x in found]:
            found.append(phrase)

    return found


def classify_topics(text: str) -> dict[str, list[str]]:
    """Rule-based topic identification with the matched keywords."""
    lower_text = text.lower()
    results = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = []
        for keyword in sorted(keywords, key=len, reverse=True):
            if keyword.lower() in lower_text and keyword not in matches:
                matches.append(keyword)
        if matches:
            results[topic] = matches

    return results


def extract_sentiment_words(text: str) -> tuple[list[str], list[str]]:
    """
    Extract positive/negative words.
    Longer phrases are checked first to avoid 'อร่อย' being extracted from
    'ไม่อร่อย' as a positive word.
    """
    lower_text = text.lower()
    positive = []
    negative = []

    # Negative phrases first.
    for phrase in sorted(NEGATIVE_WORDS, key=len, reverse=True):
        if phrase.lower() in lower_text:
            negative.append(phrase)

    # Positive matches are ignored when the phrase occurs inside a negative phrase.
    negative_spans = []
    for phrase in negative:
        start = 0
        while True:
            idx = lower_text.find(phrase.lower(), start)
            if idx == -1:
                break
            negative_spans.append((idx, idx + len(phrase)))
            start = idx + len(phrase)

    for phrase in sorted(POSITIVE_WORDS, key=len, reverse=True):
        start = 0
        while True:
            idx = lower_text.find(phrase.lower(), start)
            if idx == -1:
                break

            overlap = any(
                idx < end and idx + len(phrase) > begin
                for begin, end in negative_spans
            )
            if not overlap and phrase not in positive:
                positive.append(phrase)

            start = idx + len(phrase)

    return positive, negative


def sentiment_label(positive: list[str], negative: list[str]) -> str:
    if positive and negative:
        return "Mixed"
    if positive:
        return "Positive"
    if negative:
        return "Negative"
    return "Neutral"


# =========================================================
# Full analysis pipeline
# =========================================================

def analyze_review(original_text: str, language_choice: str) -> dict:
    metadata = extract_metadata(original_text)
    cleaned = clean_text(original_text)

    detected = detect_language(cleaned)
    if language_choice == "Auto Detect":
        language = detected
    else:
        language = language_choice

    tokens = tokenize_text(cleaned, language)
    pos_tags = pos_tag_tokens(tokens, language)
    entities = extract_entities(cleaned)
    topics = classify_topics(cleaned)
    positive, negative = extract_sentiment_words(cleaned)
    sentiment = sentiment_label(positive, negative)

    return {
        "original": original_text,
        "cleaned": cleaned,
        "metadata": metadata,
        "detected_language": detected,
        "language_used": language,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entities": entities,
        "topics": topics,
        "positive": positive,
        "negative": negative,
        "sentiment": sentiment,
    }


# =========================================================
# UI
# =========================================================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 50%); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .main-title { font-size: 2.7rem; font-weight: 900; letter-spacing: -0.04em; margin-bottom: 0.2rem; }
    .subtitle { color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .result-card { padding: 1rem; border-radius: 16px; border: 1px solid #e2e8f0; background: #ffffff; margin-bottom: 0.75rem; box-shadow: 0 6px 20px rgba(15,23,42,.06); }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem; box-shadow: 0 5px 16px rgba(15,23,42,.05); }
    [data-testid="stTextArea"] textarea { border-radius: 14px; border: 1px solid #cbd5e1; }
    .stButton > button { border-radius: 12px; font-weight: 800; min-height: 2.8rem; }
    [data-testid="stSidebar"] { background: #f8fafc; }
    .footer { text-align: center; color: #94a3b8; padding-top: 2rem; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="result-card" style="padding:1.6rem 1.8rem; background:linear-gradient(135deg,#111827,#475569); color:white;"><div style="font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; opacity:.7; font-weight:700;">NATURAL LANGUAGE PROCESSING</div><div class="main-title" style="color:white; margin-top:.35rem;">🍽️ Restaurant Review Analyzer</div><div style="opacity:.85; font-size:1rem;">ระบบวิเคราะห์และสกัดข้อมูลจากรีวิวร้านอาหารภาษาไทยและภาษาอังกฤษ</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ระบบวิเคราะห์รีวิวร้านอาหารภาษาไทยและภาษาอังกฤษด้วยเทคนิค NLP</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    language_choice = st.selectbox(
        "Review Language",
        ["Auto Detect", "Thai", "English"],
        index=0,
    )

    st.divider()
    st.markdown("### 🧩 NLP Pipeline")
    st.caption("Cleaning → Tokenization → Normalization → Topic → POS/NER → Sentiment")

    st.divider()
    st.markdown("### 💡 Test Review")
    st.caption(
        "ลองใช้รีวิวตัวอย่างจาก data/sample_reviews.txt "
        "หรือวางข้อความของคุณเองในช่องด้านซ้าย"
    )

default_review = (
    "เมื่อวานไปกินที่ร้าน MK สาขาเซ็นทรัลลาดพร้าว "
    "อาหารอร่อยมาก โดยเฉพาะสุกี้หมู น้ำซุปเข้มข้น พนักงานบริการดี "
    "แต่รออาหารนานไปหน่อย ราคาแอบแพง"
)

review = st.text_area(
    "📝 กรุณาป้อนข้อความรีวิว...",
    value=default_review,
    height=180,
    placeholder="เช่น ร้านนี้อาหารอร่อยมาก แต่บริการช้าและราคาแพง",
)

analyze_clicked = st.button("🔍 Analyze Review", type="primary", use_container_width=True)

if analyze_clicked:
    if not review.strip():
        st.warning("กรุณากรอกข้อความรีวิวก่อนวิเคราะห์")
        st.stop()

    result = analyze_review(review, language_choice)

    st.success("วิเคราะห์รีวิวเรียบร้อยแล้ว")

    # 10.1 Cleaned Text
    st.subheader("1️⃣ Text Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Review**")
        st.info(result["original"])
    with col2:
        st.markdown("**Cleaned Review**")
        st.success(result["cleaned"])

    metadata = result["metadata"]
    with st.expander("🔎 Metadata ที่ตรวจพบจากข้อความต้นฉบับ"):
        st.write("**URLs:**", metadata["urls"] or "ไม่พบ")
        st.write("**Emails:**", metadata["emails"] or "ไม่พบ")
        st.write("**Phone numbers:**", metadata["phones"] or "ไม่พบ")

    # 10.2 Tokenization
    st.subheader("2️⃣ Tokenization & Normalization")
    st.caption(
        f"Language used: **{result['language_used']}** "
        f"(Detected: **{result['detected_language']}**)"
    )
    if result["tokens"]:
        st.code(" | ".join(result["tokens"]))
    else:
        st.warning("ไม่พบ Token")

    # 10.3 POS
    st.subheader("3️⃣ POS Tagging")
    if result["pos_tags"]:
        pos_df = pd.DataFrame(result["pos_tags"], columns=["Token", "POS"])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)
    else:
        st.info("ไม่พบข้อมูลสำหรับ POS Tagging")

    # 10.4 Extracted information / NER
    st.subheader("4️⃣ Extracted Information (Hybrid NER)")
    rows = []
    for entity_type, values in result["entities"].items():
        if values:
            for value in values:
                rows.append({"ประเภทข้อมูล": entity_type, "ค่าที่พบ": value})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("ไม่พบ Entity จาก Dictionary/Regex")

    st.caption(
        "หมายเหตุ: ระบบใช้ Hybrid NER (Dictionary + Regex) เพื่อให้รองรับภาษาไทย "
        "และลดการพึ่งพาโมเดลขนาดใหญ่บน Streamlit Community Cloud"
    )

    # 10.5 Sentiment
    st.subheader("5️⃣ Sentiment Analysis")
    sentiment = result["sentiment"]
    sentiment_icon = {
        "Positive": "🟢",
        "Negative": "🔴",
        "Mixed": "🟡",
        "Neutral": "⚪",
    }[sentiment]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Sentiment", f"{sentiment_icon} {sentiment}")
    with c2:
        st.metric("Positive words", len(result["positive"]))
    with c3:
        st.metric("Negative words", len(result["negative"]))

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**คำชม / Positive**")
        st.write(", ".join(result["positive"]) if result["positive"] else "ไม่พบ")
    with s2:
        st.markdown("**คำติ / Negative**")
        st.write(", ".join(result["negative"]) if result["negative"] else "ไม่พบ")

    # 10.6 Topics
    st.subheader("6️⃣ Topic Identification")
    if result["topics"]:
        for topic, keywords in result["topics"].items():
            st.markdown(f"**• {topic}** — {', '.join(keywords)}")
    else:
        st.info("ไม่พบ Topic ที่ตรงกับ Keyword Dictionary")

    # 11. Charts
    st.subheader("7️⃣ Visualization")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Positive vs Negative**")
        sentiment_df = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Negative"],
                "Count": [len(result["positive"]), len(result["negative"])],
            }
        ).set_index("Sentiment")
        st.bar_chart(sentiment_df)

    with chart_col2:
        st.markdown("**Topic Distribution**")
        topic_df = pd.DataFrame(
            {
                "Topic": list(result["topics"].keys()),
                "Count": [len(v) for v in result["topics"].values()],
            }
        ).set_index("Topic")

        if not topic_df.empty:
            st.bar_chart(topic_df)
        else:
            st.info("ไม่มี Topic สำหรับแสดงกราฟ")

    # Summary
    st.subheader("📌 Analysis Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Entities", sum(len(v) for v in result["entities"].values()))
    summary_cols[1].metric("Tokens", len(result["tokens"]))
    summary_cols[2].metric("Topics", len(result["topics"]))
    summary_cols[3].metric("Sentiment", result["sentiment"])

else:
    st.info("พิมพ์หรือแก้ไขรีวิว แล้วกด **🔍 Analyze Review** เพื่อเริ่มวิเคราะห์")

    st.subheader("✨ ตัวอย่างความสามารถ")
    example_cols = st.columns(3)
    example_cols[0].markdown("**🏪 Entity Extraction**\n\nร้าน / แบรนด์ / สถานที่ / เมนู")
    example_cols[1].markdown("**💬 Sentiment**\n\nPositive / Negative / Mixed / Neutral")
    example_cols[2].markdown("**🏷️ Topics**\n\nอาหาร / ราคา / บริการ / รสชาติ ฯลฯ")


st.markdown('<div class="footer">Restaurant Review Analyzer • Natural Language Processing Project</div>', unsafe_allow_html=True)
