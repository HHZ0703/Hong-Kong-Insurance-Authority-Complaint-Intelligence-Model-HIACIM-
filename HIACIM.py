import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

st.set_page_config(
    page_title="IA Sentinel",
    page_icon="🛡️",
    layout="centered"
)

def main():
    st.title("🛡️ IA Sentinel")
    st.markdown("**Hong Kong Insurance Authority Complaint Analyzer**")
    st.caption("Multilingual AI Tool | 支持英文、普通話、粵語 | https://www.ia.org.hk")

    # Language selector
    language = st.selectbox(
        "Select Input Language / 選擇輸入語言",
        options=["Auto Detect", "English", "Mandarin (普通話)", "Cantonese (粵語)"],
        index=0
    )

    # Load multilingual sentiment classifier
    @st.cache_resource
    def load_classifier():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            st.info("Using multilingual sentiment model")
            return pipeline("sentiment-analysis", 
                            model="clapAI/roberta-large-multilingual-sentiment")

    classifier = load_classifier()

    # Summarizer
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")

    summarizer = load_summarizer()

    # TTS Pipelines for 3 Languages
    @st.cache_resource
    def load_tts_english():
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")

    @st.cache_resource
    def load_tts_mandarin():
        return pipeline("text-to-speech", model="facebook/mms-tts-zh")  # Mandarin Chinese

    tts_eng = load_tts_english()
    tts_zh = load_tts_mandarin()

    # Main input area
    complaint_text = st.text_area(
        "Enter Insurance Complaint (English / 中文 / 粵語)",
        height=180,
        placeholder="The insurer delayed my claim for months...\n保險公司拖延理賠三個月...\n保險公司拖咗我索償三個月都冇回覆..."
    )

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze / 分析投訴", type="primary", use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis in 3 Languages / 三語音播報", use_container_width=True)

    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing... 分析中..."):
            time.sleep(1.2)

            result = classifier(complaint_text[:512])[0]
            label = result['label'].upper()
            score = result['score']

            if "NEGATIVE" in label or score > 0.75:
                sentiment = "Negative / 負面"
                severity = "High / 高"
                color = "🔴"
                advice = "High priority – Recommend immediate IA review for possible misconduct."
            elif "POSITIVE" in label:
                sentiment = "Positive / 正面"
                severity = "Low / 低"
                color = "🟢"
                advice = "Positive feedback. Low priority."
            else:
                sentiment = "Neutral / 中性"
                severity = "Medium / 中"
                color = "🟡"
                advice = "Moderate issue. Further review recommended."

            st.success("Analysis Complete / 分析完成")
            st.markdown(f"### {color} Sentiment: **{sentiment}** (Confidence: {score:.1%})")
            st.markdown(f"### Severity: **{severity}**")
            st.info(f"**IA Recommendation / IA建議：** {advice}")

            # Summary
            st.subheader("📝 Summary / 摘要")
            summary = summarizer(complaint_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
            st.write(summary)

            # Save for TTS
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            st.session_state.last_analysis = {
                "complaint": complaint_text,
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice,
                "summary": summary
            }

    # === Generate Audio in 3 Languages ===
    if speak_btn and 'last_analysis' in st.session_state:
        analysis = st.session_state.last_analysis
        speak_text = f"""
        Analysis result: {analysis['sentiment']}. Severity level: {analysis['severity']}. 
        Summary: {analysis['summary']}. IA Recommendation: {analysis['advice']}
        """

        st.subheader("🔊 Audio Output in Three Languages")

        col_eng, col_man, col_can = st.columns(3)

        with col_eng:
            st.write("**English**")
            try:
                audio_eng = tts_eng(speak_text[:400])
                st.audio(audio_eng["audio"], sample_rate=audio_eng["sampling_rate"])
            except:
                st.error("English audio failed")

        with col_man:
            st.write("**普通話 (Mandarin)**")
            try:
                audio_zh = tts_zh(speak_text[:400])
                st.audio(audio_zh["audio"], sample_rate=audio_zh["sampling_rate"])
            except:
                st.error("Mandarin audio failed")

        with col_can:
            st.write("**粵語 (Cantonese)**")
            st.warning("⚠️ Cantonese TTS is limited on Hugging Face. Using English voice as fallback.")
            try:
                # Fallback: Use English TTS with Cantonese note
                cantonese_note = "Cantonese version: " + speak_text[:300]
                audio_can = tts_eng(cantonese_note)
                st.audio(audio_can["audio"], sample_rate=audio_can["sampling_rate"])
            except:
                st.error("Cantonese audio failed")

        st.caption("Note: Cantonese spoken synthesis has limited support. For best results, use written Chinese or English.")

    # Sidebar
    with st.sidebar:
        st.header("About IA Sentinel")
        st.write("• Supports English, Mandarin & Cantonese input")
        st.write("• Sentiment + Severity Analysis")
        st.write("• Automatic Summary")
        st.write("• Audio in English, Mandarin & Cantonese")
        st.divider()
        st.info("All models from Hugging Face.")
        st.caption("IA Sentinel | Consumer Protection Tool")

if __name__ == "__main__":
    main()
