import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
from PIL import Image

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
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception:
            st.info("Using multilingual sentiment model")
            return pipeline("sentiment-analysis", model="clapAI/roberta-large-multilingual-sentiment")

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
        placeholder="The insurer delayed my claim for 3 months...\n保險公司拖延我的索償三個月...\n保險公司拖咗我索償三個月都冇回覆...",
    )

    # Image Upload Section
    st.subheader("📸 Supporting Documents / 上傳相關文件")
    uploaded_file = st.file_uploader(
        "Upload image of policy document, email, receipt, or screenshot (optional)",
        type=["png", "jpg", "jpeg", "pdf"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type.startswith("image"):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Supporting Document", use_column_width=True)
            else:
                st.info("📄 PDF uploaded (preview not available)")
        except:
            st.warning("Unable to display uploaded file.")

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint / 分析投訴", 
                               type="primary", use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis / 語音播報", 
                             use_container_width=True)

    # ====================== ANALYSIS ======================
    if analyze_btn and complaint_text.strip():
        with st.spinner("Analyzing complaint... 分析投訴中..."):
            time.sleep(0.8)

            input_text = complaint_text[:512]
            result = classifier(input_text)[0]
            label = result['label'].upper()
            score = result['score']

            # Sentiment & Severity
            if any(x in label for x in ["NEGATIVE", "NEG"]) or score > 0.75:
                sentiment = "Negative / 負面"
                severity = "High / 高"
                color = "🔴"
                advice = "High priority – Recommend immediate IA review for possible misconduct or serious delay."
            elif any(x in label for x in ["POSITIVE", "POS"]) or score > 0.85:
                sentiment = "Positive / 正面"
                severity = "Low / 低"
                color = "🟢"
                advice = "Positive feedback or resolved issue. Low priority."
            else:
                sentiment = "Neutral / 中性"
                severity = "Medium / 中"
                color = "🟡"
                advice = "Moderate complaint. Further human review recommended."

            st.success("✅ Analysis Complete / 分析完成")

            st.markdown(f"### {color} Sentiment: **{sentiment}** (Confidence: {score:.1%})")
            st.markdown(f"### Severity: **{severity}**")
            st.info(f"**IA Recommendation / IA建議：** {advice}")

            # ====================== NEW SUGGESTION SECTION ======================
            st.subheader("💡 Suggestions / 建議行動")

            if severity == "High / 高":
                st.markdown("""
                **For the Complainant (投訴人建議):**
                - Submit a formal complaint to the Insurance Authority via the IA website or hotline (2894 1222)
                - Keep all records: policy documents, emails, call logs, and payment proofs
                - Consider contacting the **Financial Dispute Resolution Centre (FDRC)** if the dispute involves monetary claims
                """)
                
                st.markdown("""
                **For IA Staff (IA處理建議):**
                - Prioritize this case for investigation under the Insurance Ordinance
                - Request full case file from the insurer within 7 working days
                - Check for possible breaches of the **Code of Conduct for Insurers** or **Intermediaries**
                """)

            elif severity == "Medium / 中":
                st.markdown("""
                **Recommended Actions:**
                - Advise complainant to first contact the insurer’s complaint handling department in writing
                - Request a written response from the insurer within 14 days
                - If no satisfactory reply, escalate to IA with supporting documents
                - Monitor for patterns if similar complaints from the same insurer appear
                """)

            else:  # Low severity
                st.markdown("""
                **Suggested Response:**
                - Acknowledge the positive feedback to the complainant
                - No immediate IA action required
                - File the case for record and trend monitoring
                - Optionally share good practices with the insurer
                """)

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
            st.warning("Please analyze a complaint first.")
        else:
            with st.spinner("Generating audio... 生成語音中..."):
                analysis_text = f"""
                Analysis result: {st.session_state.last_analysis['sentiment']}. 
                Severity: {st.session_state.last_analysis['severity']}. 
                IA Recommendation: {st.session_state.last_analysis['advice']}
                """
                try:
                    speech = tts_pipe(analysis_text[:500])
                    st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                    st.success("🔊 Audio generated successfully")
                except Exception as e:
                    st.error(f"Audio generation failed: {str(e)}")

    # ====================== SIDEBAR ======================
    with st.sidebar:
        st.header("🛡️ About IA Sentinel")
        st.markdown("AI assistant for the Hong Kong Insurance Authority to support efficient complaint triage.")

        st.divider()
        st.subheader("Features")
        st.write("• Multilingual analysis (EN / 普通話 / 粵語)")
        st.write("• Sentiment & Severity assessment")
        st.write("• Supporting document upload")
        st.write("• Actionable suggestions")
        st.write("• Text-to-Speech")

        st.divider()
        st.info("**Disclaimer:** This tool assists with initial triage only. Final decisions rest with IA officers.")

if __name__ == "__main__":
    main()
