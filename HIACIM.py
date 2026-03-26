import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time

# Page Configuration
st.set_page_config(
    page_title="IA Sentinel",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🛡️ IA Sentinel")
    st.markdown("**Hong Kong Insurance Authority Complaint Analyzer**")
    st.caption("AI-powered multilingual sentiment & severity analysis | 支持英文、普通話、粵語 | [IA Official Site](https://www.ia.org.hk)")

    # ====================== LANGUAGE SELECTOR ======================
    language = st.selectbox(
        "Select Input Language / 選擇輸入語言",
        options=["Auto Detect", "English", "Mandarin (普通話)", "Cantonese (粵語)"],
        index=0
    )

    # ====================== MODEL LOADING ======================
    @st.cache_resource(show_spinner="Loading AI models...")
    def load_classifier():
        try:
            # Try custom fine-tuned model first
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            st.success("Loaded custom IA Complaint Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception:
            st.info("Using multilingual sentiment model (English + Chinese supported)")
            return pipeline(
                "sentiment-analysis", 
                model="clapAI/roberta-large-multilingual-sentiment"
            )

    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")

    @st.cache_resource
    def load_tts():
        # MMS-TTS supports multiple languages; fallback to English for stability
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")

    classifier = load_classifier()
    summarizer = load_summarizer()
    tts_pipe = load_tts()

    # ====================== MAIN INPUT ======================
    complaint_text = st.text_area(
        "Enter Insurance Complaint / 輸入保險投訴內容",
        height=200,
        placeholder="The insurer delayed my claim for 3 months without explanation...\n\n保險公司拖延我的索償三個月，至今沒有回覆...\n\n保險公司拖咗我索償三個月都冇回覆...",
        help="Supports English, Mandarin, and Cantonese"
    )

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint / 分析投訴", 
                               type="primary", 
                               use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis / 語音播報", 
                             use_container_width=True)

    # ====================== ANALYSIS LOGIC ======================
    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing complaint... 分析投訴中..."):
            time.sleep(0.8)

            # Truncate to model's max length
            input_text = complaint_text[:512]

            # Classification
            result = classifier(input_text)[0]
            label = result['label'].upper()
            score = result['score']

            # Enhanced sentiment & severity mapping
            if any(x in label for x in ["NEGATIVE", "NEG"]) or score > 0.75:
                sentiment = "Negative / 負面"
                severity = "High / 高"
                color = "🔴"
                advice = "High priority – Recommend immediate IA review. Possible misconduct, unreasonable delay, or serious intermediary conduct issue."
            elif any(x in label for x in ["POSITIVE", "POS"]) or score > 0.85:
                sentiment = "Positive / 正面"
                severity = "Low / 低"
                color = "🟢"
                advice = "Positive feedback or resolved issue. Low priority."
            else:
                sentiment = "Neutral / 中性"
                severity = "Medium / 中"
                color = "🟡"
                advice = "Moderate complaint. Further human review is recommended."

            st.success("✅ Analysis Complete / 分析完成")

            st.markdown(f"### {color} Sentiment: **{sentiment}** (Confidence: {score:.1%})")
            st.markdown(f"### Severity: **{severity}**")
            st.info(f"**IA Recommendation / IA建議：** {advice}")

            # Summary
            st.subheader("📝 Summary / 摘要")
            try:
                summary_output = summarizer(
                    complaint_text, 
                    max_length=130, 
                    min_length=40, 
                    do_sample=False
                )[0]['summary_text']
                st.write(summary_output)
            except:
                st.write("Summary generation failed. Displaying original text excerpt.")
                st.write(complaint_text[:400] + "...")

            # Store results for TTS
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            
            st.session_state.last_analysis = {
                "text": complaint_text,
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice,
                "summary": summary_output if 'summary_output' in locals() else complaint_text[:300]
            }

    # ====================== TEXT-TO-SPEECH ======================
    if speak_btn:
        if 'last_analysis' not in st.session_state:
            st.warning("Please analyze a complaint first before using text-to-speech.")
        else:
            with st.spinner("Generating audio... 生成語音中..."):
                analysis_text = f"""
                Analysis result for Insurance Authority. 
                Sentiment: {st.session_state.last_analysis['sentiment']}. 
                Severity: {st.session_state.last_analysis['severity']}. 
                Summary: {st.session_state.last_analysis['summary']}. 
                Recommendation: {st.session_state.last_analysis['advice']}
                """

                try:
                    speech = tts_pipe(analysis_text[:450])
                    st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                    st.success("🔊 Audio generated successfully (English voice)")
                    st.caption("Note: Cantonese TTS is currently limited. English and Mandarin perform best.")
                except Exception as e:
                    st.error(f"Audio generation failed: {str(e)}")

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("🛡️ About IA Sentinel")
        st.markdown("""
        Multilingual AI assistant designed for the **Hong Kong Insurance Authority**  
        to support efficient complaint triage and consumer protection.
        """)

        st.divider()
        st.subheader("Features")
        st.write("• Multilingual support (English / 普通話 / 粵語)")
        st.write("• Sentiment Analysis")
        st.write("• Severity Assessment")
        st.write("• Automatic Complaint Summary")
        st.write("• Text-to-Speech readout")
        
        st.divider()
        st.info("""
        **Note to IA Staff:**  
        This tool assists with initial triage. 
        All final decisions should be made by qualified IA officers.
        """)
        
        st.caption("IA Sentinel | Powered by Hugging Face Transformers")

if __name__ == "__main__":
    main()
