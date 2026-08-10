# 🍽️ Restaurant Review Analyzer

ระบบวิเคราะห์และสกัดข้อมูลจากรีวิวร้านอาหารภาษาไทยและภาษาอังกฤษด้วยเทคนิค Natural Language Processing (NLP) พัฒนาด้วย Python และ Streamlit

## 1. Project Name

**Restaurant Review Analyzer**

## 2. Project Description

Web Application สำหรับรับข้อความรีวิวร้านอาหาร แล้วทำการทำความสะอาดข้อความ ตัดคำ Normalize ข้อความ วิเคราะห์ Topic สกัด Entity สกัดคำชม/คำติ และวิเคราะห์ Sentiment

ระบบออกแบบให้เหมาะกับงานรายวิชา Natural Language Processing และเน้นการทำงานแบบ Lightweight เพื่อให้สามารถ Deploy บน Streamlit Community Cloud ได้โดยไม่ต้องใช้ API Key หรือบริการแบบเสียเงิน

## 3. Objectives

- วิเคราะห์รีวิวภาษาไทยและภาษาอังกฤษ
- แสดงขั้นตอน NLP ตั้งแต่ Cleaning จนถึงผลลัพธ์
- สกัดชื่อร้าน/แบรนด์ สถานที่ และเมนู
- จำแนก Topic ของรีวิว
- สกัดคำชมและคำติ
- วิเคราะห์ Sentiment เป็น Positive, Negative, Mixed หรือ Neutral
- แสดงผลในรูปแบบ Dashboard และกราฟ

## 4. Features

- Text Cleaning ด้วย Regex
- URL / Email / Phone Metadata Extraction
- Thai Tokenization ด้วย PyThaiNLP
- English Tokenization ด้วย Regex
- Thai Text Normalization ด้วย PyThaiNLP
- POS Tagging
- Hybrid NER ด้วย Dictionary + Regex
- Rule-based Topic Identification
- Rule-based Sentiment Analysis
- Positive / Negative Keyword Extraction
- Dashboard และ Bar Chart
- รองรับ Thai / English / Auto Detect

## 5. NLP Techniques

### Regex & Cleansing

ใช้ Regex เพื่อ:

- ลบ URL
- ตรวจจับ Email
- ตรวจจับเบอร์โทรศัพท์
- ลบอักขระที่ไม่จำเป็น
- ลดช่องว่างซ้ำ
- ลดการพิมพ์ตัวอักษรซ้ำ เช่น `อร่อยมากกกก` → `อร่อยมาก`

ข้อมูล URL, Email และเบอร์โทรศัพท์จะถูกเก็บใน Metadata ก่อนลบออกจาก Cleaned Text

### Tokenization & Normalization

- ภาษาไทยใช้ `PyThaiNLP` และ `newmm`
- ภาษาอังกฤษใช้ Regex tokenizer และแปลงเป็น lowercase
- ภาษาอังกฤษมี Stopword ขนาดเล็กในตัวโปรแกรมเพื่อไม่ต้องติดตั้ง NLTK
- ใช้ `pythainlp.normalize()` สำหรับ Normalize ข้อความภาษาไทย

### Topic Identification

ใช้ Keyword Dictionary แบบ Rule-based จำแนก:

- อาหาร
- รสชาติ
- ราคา
- การบริการ
- สถานที่
- ความสะอาด
- บรรยากาศ
- การจัดส่ง

ระบบจะแสดง Topic และ Keyword ที่เป็นเหตุผลในการจำแนก

### POS & NER

**POS**

- ภาษาไทยใช้ PyThaiNLP POS Tagging
- ภาษาอังกฤษใช้ Lightweight Rule-based POS fallback เพื่อลดขนาด Dependency

**NER**

ระบบใช้ Hybrid Approach:

```text
NER / Entity Extraction
        ├── Dictionary
        └── Regex
```

Dictionary ใช้สำหรับชื่อแบรนด์ สถานที่ และเมนูที่พบบ่อย ส่วน Regex ใช้ช่วยตรวจจับรูปแบบ เช่น `สาขา...`

เหตุผลที่ใช้ Hybrid Approach คือโมเดล NER ภาษาไทยที่มีความแม่นยำสูงอาจมี Dependency และขนาดใหญ่ขึ้น ซึ่งไม่จำเป็นสำหรับงานสาธิตนี้ และอาจทำให้ Deployment บน Streamlit Community Cloud ซับซ้อนขึ้น

### Sentiment Analysis

ใช้ Positive/Negative Dictionary แบบ Rule-based

ผลลัพธ์:

- Positive
- Negative
- Mixed
- Neutral

หากพบทั้งคำ Positive และ Negative จะจัดเป็น `Mixed`

## 6. NLP Pipeline

```text
User Review
     ↓
Metadata Extraction
     ↓
Text Cleaning / Regex
     ↓
Normalization
     ↓
Tokenization
     ↓
POS Tagging
     ↓
Hybrid Entity Extraction
     ↓
Topic Identification
     ↓
Positive / Negative Extraction
     ↓
Sentiment Classification
     ↓
Dashboard + Visualization
```

## 7. System Architecture

```text
┌─────────────────────────────┐
│       Streamlit UI          │
│ Text Area + Language Select │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│     Preprocessing Layer     │
│ Regex + Cleaning + Normalize│
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│       NLP Analysis          │
│ Tokenize + POS + Hybrid NER │
│ Topic + Sentiment           │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│      Result Dashboard       │
│ Tables + Metrics + Charts   │
└─────────────────────────────┘
```

## 8. Installation

แนะนำ Python 3.10–3.12

```bash
git clone https://github.com/<YOUR_USERNAME>/restaurant-review-analyzer.git
cd restaurant-review-analyzer

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

ติดตั้ง Library:

```bash
pip install -r requirements.txt
```

## 9. How to Run

```bash
streamlit run app.py
```

จากนั้นเปิด URL ที่ Streamlit แสดงใน Terminal เช่น:

```text
http://localhost:8501
```

## 10. Example Input

```text
เมื่อวานไปกินที่ร้าน MK สาขาเซ็นทรัลลาดพร้าว
อาหารอร่อยมาก โดยเฉพาะสุกี้หมู น้ำซุปเข้มข้น
พนักงานบริการดี แต่รออาหารนานไปหน่อย ราคาแพง
```

English example:

```text
KFC at Central World is great.
The fried chicken is crispy and delicious.
The staff are friendly, but the restaurant is quite crowded.
```

## 11. Example Output

ตัวอย่างผลลัพธ์:

```text
Brand / Restaurant:
MK

Location:
เซ็นทรัลลาดพร้าว

Menu / Food:
สุกี้หมู
น้ำซุป

Positive:
อร่อยมาก
เข้มข้น
บริการดี

Negative:
รอนาน
แพง

Sentiment:
Mixed

Topics:
อาหาร
รสชาติ
การบริการ
ราคา
สถานที่
```

## 12. Project Structure

```text
restaurant-review-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── sample_reviews.txt
```

## 13. Deployment

สามารถ Deploy บน **Streamlit Community Cloud** โดยเชื่อมต่อกับ GitHub Repository

ขั้นตอน:

1. สร้าง GitHub Repository ชื่อ `restaurant-review-analyzer`
2. Upload ไฟล์ทั้งหมดตาม Project Structure
3. ตรวจสอบว่า `app.py` และ `requirements.txt` อยู่ที่ root ของ Repository
4. เข้า Streamlit Community Cloud
5. เชื่อม GitHub Account
6. กด `Create app`
7. เลือก Repository
8. เลือก Branch เช่น `main`
9. เลือก Main file path เป็น `app.py`
10. กด `Deploy`

Community Cloud จะติดตั้ง Dependency จาก `requirements.txt` และสร้าง URL สำหรับใช้งานจริง

> URL สำหรับส่งอาจารย์จะเป็น URL ที่ Streamlit สร้างให้ เช่น `https://your-app-name.streamlit.app`

## 14. GitHub Repository

หลังสร้าง Repository ให้ใส่ URL จริงของผู้พัฒนา เช่น:

```text
https://github.com/<YOUR_USERNAME>/restaurant-review-analyzer
```

## 15. Streamlit URL

หลัง Deploy สำเร็จ ให้ใส่ URL จริง เช่น:

```text
https://<YOUR_APP_NAME>.streamlit.app
```

## 16. Limitations

- NER เป็น Hybrid Rule-based ไม่ใช่ Transformer NER
- Entity ที่เป็นชื่อร้าน/เมนูใหม่ ๆ ซึ่งไม่มีใน Dictionary อาจตรวจจับไม่ได้
- Topic Classification เป็น Keyword-based จึงอาจไม่เข้าใจบริบทซับซ้อน
- Sentiment เป็น Lexicon/Rule-based ไม่ได้ใช้ Machine Learning
- การตรวจจับภาษาเป็น heuristic จากจำนวนตัวอักษร
- English POS เป็น Lightweight Rule-based
- รีวิวที่มี sarcasm, slang หรือบริบทซับซ้อนอาจวิเคราะห์ผิด
- ระบบนี้เหมาะกับงานเรียนและการสาธิต NLP มากกว่าการใช้งาน Production

## 17. AI Prompt

Prompt ที่ใช้ช่วยพัฒนา Web Application:

```text
คุณคือผู้เชี่ยวชาญด้าน Python, NLP และ Streamlit
ช่วยพัฒนา Web Application สำหรับงานวิชา Natural Language Processing (NLP)
หัวข้อ “ระบบวิเคราะห์และสกัดข้อมูลจากรีวิวร้านอาหาร (Restaurant Review Analyzer)”

ระบบต้องรองรับข้อความรีวิวภาษาไทยและภาษาอังกฤษ โดยผู้ใช้สามารถป้อนข้อความรีวิว
และระบบต้องทำ NLP Pipeline ดังนี้:

User Review
→ Text Cleaning
→ Tokenization
→ Normalization
→ Topic Identification
→ POS / NER
→ Keyword & Sentiment Extraction
→ Result Dashboard

ต้องใช้ Regex สำหรับ URL, Email, เบอร์โทรศัพท์, อักขระพิเศษ,
ช่องว่างซ้ำ และข้อความที่มีการพิมพ์ตัวอักษรซ้ำ

ภาษาไทยใช้ PyThaiNLP สำหรับ Tokenization และ Normalization
ภาษาอังกฤษต้อง lowercase, tokenize และจัดการ stopwords

ระบบต้องสกัด:
- ชื่อร้าน / แบรนด์
- สถานที่
- เมนูอาหาร
- คำชม
- คำติ
- Sentiment
- Topic

Sentiment มี Positive, Negative, Mixed และ Neutral
Topic มี อาหาร, รสชาติ, ราคา, การบริการ, สถานที่, ความสะอาด,
บรรยากาศ, การจัดส่ง และอื่น ๆ

หาก NER ภาษาไทยไม่แม่นยำ ให้ใช้ Hybrid Approach:
Dictionary + Regex และอธิบายเหตุผลไว้ใน README

สร้าง Streamlit Dashboard ที่มี:
- Original Review
- Cleaned Review
- Tokenization
- POS Tagging
- Extracted Information
- Sentiment Metrics
- Positive / Negative words
- Topic + Keywords
- Positive vs Negative Bar Chart
- Topic Distribution Bar Chart

สร้างไฟล์:
app.py
requirements.txt
README.md
data/sample_reviews.txt
.gitignore

ต้องไม่ใช้ API Key หรือบริการเสียเงิน
และต้อง Deploy ได้บน Streamlit Community Cloud
เขียนโค้ดให้เหมาะกับนักศึกษา แบ่ง Function ชัดเจน
และมี Error Handling เมื่อผู้ใช้ไม่กรอกข้อความหรือไม่พบ Entity
```

## 18. Testing Checklist

ก่อนส่งงานให้ทดสอบ:

- [ ] `pip install -r requirements.txt` ผ่าน
- [ ] `streamlit run app.py` เปิดได้
- [ ] รีวิวภาษาไทยวิเคราะห์ได้
- [ ] รีวิวภาษาอังกฤษวิเคราะห์ได้
- [ ] Auto Detect ทำงาน
- [ ] URL/Email/Phone ถูกแยกเป็น Metadata
- [ ] Token แสดงผล
- [ ] POS แสดงผล
- [ ] Entity แสดงผล
- [ ] Topic แสดงผล
- [ ] Sentiment แสดงผล
- [ ] กราฟ Positive/Negative แสดงผล
- [ ] กราฟ Topic แสดงผล
- [ ] Push ขึ้น GitHub แล้ว
- [ ] Deploy บน Streamlit Community Cloud สำเร็จ
- [ ] เปิด URL จากเครื่องอื่นได้
