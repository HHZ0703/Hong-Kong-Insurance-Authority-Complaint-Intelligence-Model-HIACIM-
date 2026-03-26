import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

# Page Configuration
st.set_page_config(
    page_title="IA Sentinel",
    page_icon="🛡️",
    layout="centered"
)

def main():
    st.title("🛡️ IA Sentinel")
    st.markdown("**Hong Kong Insurance Authority Complaint Analyzer**")
    st.caption("AI-powered sentiment & severity analysis for insurance complaints | https://www.ia.org.hk")

    # Load the fine-tuned model (fallback to general sentiment if not found)
    @st.cache_resource
    def load_model():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            st.warning("Fine-tuned model not found. Using general sentiment pipeline.")
            return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    classifier = load_model()

    # Extra pipelines (all from Hugging Face)
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")

    @st.cache_resource
    def load_tts():
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")

    summarizer = load_summarizer()
    tts_pipe = load_tts()

    # Main Input
    complaint_text = st.text_area(
        "Enter Insurance Complaint Text",
        height=180,
        placeholder="The insurance company delayed my claim settlement for months and refused to provide any explanation..."
    )

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint", type="primary", use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis", use_container_width=True)

    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing complaint..."):
            time.sleep(1.2)
            
            # Main Classification
            results = classifier(complaint_text[:512])[0]
            label = results['label']
            score = results['score']

            # Map to IA-friendly output
            if "NEGATIVE" in label.upper() or score > 0.75:
                sentiment = "Negative"
                severity = "High"
                color = "🔴"
                advice = "High priority – Possible insurer misconduct or serious delay. Recommend immediate IA review."
            elif "POSITIVE" in label.upper():
                sentiment = "Positive"
                severity = "Low"
                color = "🟢"
                advice = "Positive feedback or resolved issue. Low priority."
            else:
                sentiment = "Neutral"
                severity = "Medium"
                color = "🟡"
                advice = "Moderate concern. Further human review recommended."

            # Display Results
            st.success("Analysis Complete")
            st.markdown(f"### {color} Sentiment: **{sentiment}** (Confidence: {score:.1%})")
            st.markdown(f"### Severity Level: **{severity}**")
            st.info(f"**IA Recommendation:** {advice}")

            # Extra Features
            st.subheader("📝 Complaint Summary")
            summary = summarizer(complaint_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            st.write(summary)

            # Add to session state for audio
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            st.session_state.last_analysis = {
                "text": complaint_text,
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice,
                "summary": summary
            }

    # Text-to-Speech Feature
    if speak_btn and 'last_analysis' in st.session_state:
        with st.spinner("Generating audio..."):
            analysis_text = f"""
            Complaint analysis: {st.session_state.last_analysis['sentiment']} sentiment, 
            {st.session_state.last_analysis['severity']} severity. 
            Summary: {st.session_state.last_analysis['summary']}
            Recommendation: {st.session_state.last_analysis['advice']}
            """
            try:
                speech_output = tts_pipe(analysis_text[:500])
                st.audio(speech_output["audio"], sample_rate=speech_output["sampling_rate"])
                st.success("🔊 Audio generated successfully!")
            except:
                st.error("Audio generation failed. Try again.")

    # Sidebar Information
    with st.sidebar:
        st.header("About IA Sentinel")
        st.write("AI tool to help the Hong Kong Insurance Authority triage complaints efficiently.")
        st.write("• Sentiment Analysis")
        st.write("• Severity Assessment")
        st.write("• Automatic Summary")
        st.write("• Text-to-Speech")
        
        st.divider()
        st.info("All models and datasets are from Hugging Face.")
        st.caption("Built for regulatory compliance and consumer protection.")

    # Footer
    st.caption("IA Sentinel | Fine-tuned on financial sentiment data | Hugging Face + Streamlit")

if __name__ == "__main__":
    main()
