import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from datetime import datetime

# ==============================================
# IA Sentinel - Full Creative Application
# All pipelines & models from Hugging Face only
# ==============================================

st.set_page_config(
    page_title="IA Sentinel",
    page_icon="🛡️",
    layout="wide"
)

def main():
    st.title("🛡️ IA Sentinel")
    st.markdown("**Hong Kong Insurance Authority Intelligent Complaint Analyzer**")
    st.caption("AI-powered platform for complaint triage, consumer protection & regulatory efficiency | https://www.ia.org.hk/en/index.html")

    # Cache all Hugging Face pipelines (only once)
    @st.cache_resource
    def load_pipelines():
        try:
            # Core fine-tuned model (from your Colab training)
            classifier = pipeline(
                "text-classification",
                model="IA_Complaint_Classifier",
                tokenizer="IA_Complaint_Classifier"
            )
        except:
            classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

        return {
            "classifier": classifier,
            "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
            "ner": pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple"),
            "qa": pipeline("question-answering", model="distilbert-base-cased-distilled-squad"),
            "translator": pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh"),
            "generator": pipeline("text-generation", model="distilbert/distilgpt2"),
            "tts": pipeline("text-to-speech", model="facebook/mms-tts-eng")
        }

    pipes = load_pipelines()

    # Sidebar - IA Information & Quick Links
    with st.sidebar:
        st.header("🛡️ IA Sentinel")
        st.write("**Functions included (all from Hugging Face):**")
        st.write("• Sentiment & Severity Analysis")
        st.write("• Automatic Summarization")
        st.write("• Named Entity Recognition")
        st.write("• Question Answering")
        st.write("• English → Traditional Chinese Translation")
        st.write("• Smart Reply Generator")
        st.write("• Text-to-Speech Audio")
        st.divider()
        st.info("Tailored for Hong Kong Insurance Authority complaint handling and consumer protection.")
        st.caption("All models & datasets strictly from Hugging Face.")

    # Main Tabs - Creative & Useful Functions
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Analyze Complaint", 
        "📝 Summary & Entities", 
        "💬 Smart Reply Generator", 
        "🌐 Translate to Chinese", 
        "❓ Ask About Complaint", 
        "🔊 Text-to-Speech"
    ])

    complaint_text = st.text_area(
        "Enter the insurance complaint (English)",
        height=160,
        placeholder="The insurance company delayed my claim settlement for 3 months without explanation and refused to refund the premium...",
        key="main_input"
    )

    # ===================== TAB 1: Core Sentiment & Severity =====================
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Analyze Complaint", type="primary", use_container_width=True):
                if complaint_text.strip():
                    with st.spinner("Analyzing with fine-tuned model..."):
                        time.sleep(1)
                        result = pipes["classifier"](complaint_text[:512])[0]
                        label = result['label']
                        score = result['score']

                        if "NEGATIVE" in label.upper() or score > 0.75:
                            sentiment, severity, color, advice = "Negative", "High", "🔴", "Urgent IA review recommended – possible misconduct or serious delay."
                        elif "POSITIVE" in label.upper():
                            sentiment, severity, color, advice = "Positive", "Low", "🟢", "Positive feedback. Low priority."
                        else:
                            sentiment, severity, color, advice = "Neutral", "Medium", "🟡", "Moderate concern. Further human review suggested."

                        st.success("✅ Analysis Complete")
                        st.markdown(f"### {color} Sentiment: **{sentiment}** ({score:.1%})")
                        st.markdown(f"### Severity: **{severity}**")
                        st.info(f"**IA Recommendation:** {advice}")
                else:
                    st.warning("Please enter complaint text.")

    # ===================== TAB 2: Summary & Named Entities =====================
    with tab2:
        if st.button("📋 Generate Summary & Extract Entities", use_container_width=True):
            if complaint_text:
                with st.spinner("Summarizing and extracting key information..."):
                    summary = pipes["summarizer"](complaint_text, max_length=120, min_length=30)[0]['summary_text']
                    entities = pipes["ner"](complaint_text)

                    st.subheader("📝 Summary")
                    st.write(summary)

                    st.subheader("🔖 Key Entities Detected")
                    for ent in entities[:8]:  # limit display
                        st.write(f"• **{ent['word']}** → {ent['entity_group']} (score: {ent['score']:.3f})")
            else:
                st.warning("No text to analyze.")

    # ===================== TAB 3: Smart Reply Generator =====================
    with tab3:
        if st.button("✉️ Generate Professional Reply / Acknowledgment Letter", use_container_width=True):
            if complaint_text:
                with st.spinner("Generating polite official response..."):
                    prompt = f"Write a professional, empathetic reply from the Hong Kong Insurance Authority to this complaint: {complaint_text[:300]}"
                    generated = pipes["generator"](prompt, max_length=250, num_return_sequences=1, temperature=0.7)[0]['generated_text']
                    st.subheader("📨 Suggested Official Reply")
                    st.write(generated)
                    st.download_button("Download as .txt", generated, file_name=f"IA_Reply_{datetime.now().strftime('%Y%m%d')}.txt")
            else:
                st.warning("Enter complaint first.")

    # ===================== TAB 4: Translation to Traditional Chinese =====================
    with tab4:
        if st.button("🇭🇰 Translate to Traditional Chinese", use_container_width=True):
            if complaint_text:
                with st.spinner("Translating..."):
                    translated = pipes["translator"](complaint_text[:512])[0]['translation_text']
                    st.subheader("🇭🇰 Traditional Chinese Translation")
                    st.write(translated)
                    st.download_button("Download Translation", translated, file_name="translated_complaint.txt")
            else:
                st.warning("No text provided.")

    # ===================== TAB 5: Question Answering =====================
    with tab5:
        question = st.text_input("Ask a question about this complaint (e.g., What is the main issue?)")
        if st.button("❓ Get Answer") and question and complaint_text:
            with st.spinner("Finding answer..."):
                qa_result = pipes["qa"](question=question, context=complaint_text[:1024])
                st.success(f"**Answer:** {qa_result['answer']}")
                st.caption(f"Confidence: {qa_result['score']:.1%}")

    # ===================== TAB 6: Text-to-Speech =====================
    with tab6:
        st.write("Convert analysis or full complaint into spoken audio")
        if st.button("🔊 Speak the Full Analysis"):
            if 'last_result' not in st.session_state and complaint_text:
                # Run quick analysis first
                result = pipes["classifier"](complaint_text[:512])[0]
                label = result['label']
                sent = "Negative" if "NEGATIVE" in label.upper() else "Positive" if "POSITIVE" in label.upper() else "Neutral"
                tts_text = f"Complaint analysis: {sent} sentiment. Summary: {complaint_text[:200]}"
            else:
                tts_text = complaint_text[:500]

            with st.spinner("Generating audio..."):
                try:
                    speech = pipes["tts"](tts_text)
                    st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                    st.success("✅ Audio ready!")
                except:
                    st.error("TTS temporarily unavailable.")

        # Bonus: Download full report button
        if complaint_text and st.button("📄 Download Complete IA Report"):
            report = f"""IA Sentinel Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}
Complaint: {complaint_text}
Analysis: Generated using Hugging Face models.
All functions performed on {datetime.now().strftime('%Y-%m-%d')}"""
            st.download_button("Download Full Report (.txt)", report, file_name="IA_Sentinel_Report.txt")

    st.caption("IA Sentinel | All models & pipelines from Hugging Face | Fine-tuned DistilBERT + 7 additional HF pipelines")

if __name__ == "__main__":
    main()
