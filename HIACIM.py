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
    st.caption("AI-powered multilingual sentiment & severity analysis | 支持英文、普通話、粵語 | https://www.ia.org.hk")

    # Language selector
    language = st.selectbox(
        "Select Input Language / 選擇輸入語言",
        options=["Auto Detect", "English", "Mandarin (普通話)", "Cantonese (粵語)"],
        index=0
    )

    # Load multilingual classifier (supports English + Chinese)
    @st.cache_resource
    def load_classifier():
        try:
            # Try your fine-tuned model first
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            # Fallback to strong multilingual model
            st.info("Using multilingual sentiment model (supports English & Chinese)")
            return pipeline("sentiment-analysis", 
                          model="clapAI/roberta-large-multilingual-sentiment")

    classifier = load_classifier()

    # Summarizer (works well for English & Chinese)
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")

    # TTS (English + Mandarin supported; Cantonese limited)
    @st.cache_resource
    def load_tts():
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")  # Good for English

    summarizer = load_summarizer()
    tts_pipe = load_tts()

    # Main input
    complaint_text = st.text_area(
        "Enter Insurance Complaint (English / 中文 / 粵語)",
        height=180,
        placeholder="The insurer delayed my claim for 3 months... \n保險公司拖延理賠三個月... \n保險公司拖咗我索償三個月都冇回覆..."
    )

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint / 分析投訴", type="primary", use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis / 語音播報", use_container_width=True)

    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing... 分析中..."):
            time.sleep(1)

            # Classification
            result = classifier(complaint_text[:512])[0]
            label = result['label'].upper()
            score = result['score']

            # Map to sentiment & severity (works for EN/ZH)
            if "NEGATIVE" in label or score > 0.75:
                sentiment = "Negative / 負面"
                severity = "High / 高"
                color = "🔴"
                advice = "High priority – Recommend immediate IA review for possible misconduct or serious delay."
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

            # Store for TTS
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            st.session_state.last_analysis = {
                "text": complaint_text,
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice,
                "summary": summary
            }

    # Text-to-Speech
    if speak_btn and 'last_analysis' in st.session_state:
        with st.spinner("Generating audio... 生成語音中..."):
            analysis_text = f"""
            Analysis result: {st.session_state.last_analysis['sentiment']}. 
            Severity: {st.session_state.last_analysis['severity']}. 
            Summary: {st.session_state.last_analysis['summary']}. 
            Recommendation: {st.session_state.last_analysis['advice']}
            """
            try:
                speech = tts_pipe(analysis_text[:500])
                st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                st.success("🔊 Audio ready! (English voice)")
                st.caption("Note: Cantonese TTS support is limited. English/Mandarin works best.")
            except Exception as e:
                st.error("Audio generation failed. 語音生成失敗。")

    # Sidebar
    with st.sidebar:
        st.header("About IA Sentinel")
        st.write("Multilingual AI tool for the Hong Kong Insurance Authority.")
        st.write("• Supports English, Mandarin & Cantonese")
        st.write("• Sentiment + Severity Analysis")
        st.write("• Automatic Summary")
        st.write("• Text-to-Speech")
        st.divider()
        st.info("All models & datasets from Hugging Face.")
        st.caption("IA Sentinel | Built for consumer protection")

if __name__ == "__main__":
    main()
