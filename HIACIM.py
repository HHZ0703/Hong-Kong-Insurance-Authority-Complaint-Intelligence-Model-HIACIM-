import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
from PIL import Image
import io

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

    # Language Selector
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
            st.success("✅ Loaded custom IA Complaint Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception:
            st.info("Using multilingual sentiment model (supports English & Chinese)")
            return pipeline(
                "sentiment-analysis", 
                model="clapAI/roberta-large-multilingual-sentiment"
            )

    @st.cache_resource
    def load_tts():
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")

    classifier = load_classifier()
    tts_pipe = load_tts()

    # ====================== INPUT SECTION ======================
    st.subheader("📝 Complaint Text")
    complaint_text = st.text_area(
        "Enter Insurance Complaint / 輸入保險投訴內容",
        height=180,
        placeholder="The insurer delayed my claim for 3 months...\n\n保險公司拖延我的索償三個月...\n\n保險公司拖咗我索償三個月都冇回覆...",
        help="Supports English, Mandarin, and Cantonese"
    )

    # ====================== IMAGE UPLOAD (Replaces Summary) ======================
    st.subheader("📸 Supporting Documents / 上傳相關文件")
    uploaded_file = st.file_uploader(
        "Upload image of policy document, email, receipt, or screenshot (optional)",
        type=["png", "jpg", "jpeg", "pdf"],
        help="You can upload supporting evidence such as policy documents, correspondence, or claim rejection letters."
    )

    # Display uploaded image
    if uploaded_file is not None:
        try:
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Supporting Document", use_column_width=True)
            else:
                st.info("📄 PDF file uploaded. Image preview not available.")
        except Exception:
            st.warning("Could not display the uploaded file.")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint / 分析投訴", 
                               type="primary", 
                               use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis / 語音播報", 
                             use_container_width=True)

    # ====================== ANALYSIS ======================
    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing complaint... 分析投訴中..."):
            time.sleep(0.8)

            input_text = complaint_text[:512]

            # Classification
            result = classifier(input_text)[0]
            label = result['label'].upper()
            score = result['score']

            # Sentiment & Severity Mapping
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

            # Store for TTS
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            
            st.session_state.last_analysis = {
                "text": complaint_text,
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice,
                "has_image": uploaded_file is not None
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
                Recommendation: {st.session_state.last_analysis['advice']}
                """

                try:
                    speech = tts_pipe(analysis_text[:500])
                    st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                    st.success("🔊 Audio generated successfully")
                    st.caption("Note: English voice used. Cantonese support is limited.")
                except Exception as e:
                    st.error(f"Audio generation failed: {str(e)}")

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("🛡️ About IA Sentinel")
        st.markdown("""
        AI assistant for the **Hong Kong Insurance Authority**  
        to support efficient complaint triage and consumer protection.
        """)

        st.divider()
        st.subheader("Features")
        st.write("• Multilingual complaint analysis")
        st.write("• Sentiment & Severity assessment")
        st.write("• Image / Document upload support")
        st.write("• Text-to-Speech readout")
        
        st.divider()
        st.info("""
        **For IA Staff:**  
        This tool is for initial triage only. 
        Final decisions must be made by qualified officers.
        """)
        
        st.caption("IA Sentinel | Built for consumer protection in Hong Kong")

if __name__ == "__main__":
    main()
