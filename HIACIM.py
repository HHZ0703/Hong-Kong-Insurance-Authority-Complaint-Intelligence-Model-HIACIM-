import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
from PIL import Image
import easyocr
import torch

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
    st.caption("AI-powered multilingual sentiment & severity analysis + OCR | 支持英文、普通話、粵語")

    # ====================== MODEL LOADING ======================
    @st.cache_resource(show_spinner="Loading AI models...")
    def load_classifier():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except Exception:
            st.info("Using fallback multilingual sentiment model")
            return pipeline("sentiment-analysis", model="clapAI/roberta-large-multilingual-sentiment")

    @st.cache_resource
    def load_tts():
        return pipeline("text-to-speech", model="facebook/mms-tts-eng")

    classifier = load_classifier()
    tts_pipe = load_tts()

    # ====================== OCR READER (cached) ======================
    @st.cache_resource
    def load_ocr_reader():
        st.info("Loading OCR model... (first time may take 10-30 seconds)")
        return easyocr.Reader(['en', 'ch_sim'], gpu=torch.cuda.is_available())  # English + Simplified Chinese

    ocr_reader = load_ocr_reader()

    # ====================== INPUT SECTION ======================
    st.subheader("📝 Complaint Text")
    complaint_text = st.text_area(
        "Enter Insurance Complaint / 輸入保險投訴內容",
        height=150,
        placeholder="The insurer delayed my claim for 3 months...\n保險公司拖延我的索償三個月...\n保險公司拖咗我索償三個月都冇回覆...",
    )

    # Image Upload + OCR
    st.subheader("📸 Supporting Documents / 上傳相關文件 (OCR Enabled)")
    uploaded_file = st.file_uploader(
        "Upload image (policy, claim letter, rejection notice, receipt, etc.)",
        type=["png", "jpg", "jpeg"],
    )

    ocr_extracted_text = ""

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Document", use_column_width=True)

            # Run OCR
            with st.spinner("Running OCR on the image... 正在進行文字識別..."):
                ocr_result = ocr_reader.readtext(np.array(image), detail=0)  # detail=0 returns only text
                ocr_extracted_text = " ".join(ocr_result)

            if ocr_extracted_text.strip():
                st.success("✅ OCR Text Extracted")
                with st.expander("View Extracted Text from Image"):
                    st.write(ocr_extracted_text)
            else:
                st.warning("No text detected in the image.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Combine complaint text + OCR text for analysis
    final_text_for_analysis = complaint_text.strip()
    if ocr_extracted_text.strip():
        final_text_for_analysis += "\n\n[Document Content]\n" + ocr_extracted_text

    col1, col2 = st.columns(2)
    with col1:
        analyze_btn = st.button("🔍 Analyze Complaint / 分析投訴", 
                               type="primary", use_container_width=True)
    with col2:
        speak_btn = st.button("🔊 Speak Analysis / 語音播報", 
                             use_container_width=True)

    # ====================== ANALYSIS ======================
    if analyze_btn and final_text_for_analysis.strip():
        with st.spinner("Analyzing complaint... 分析投訴中..."):
            time.sleep(0.8)

            input_text = final_text_for_analysis[:512]
            result = classifier(input_text)[0]
            label = result['label'].upper()
            score = result['score']

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

            # Suggestions Section
            st.subheader("💡 Suggestions / 建議行動")
            if severity == "High / 高":
                st.markdown("**For Complainant:** Submit formal complaint to IA (2894 1222) and keep all records.\n**For IA Staff:** Prioritize investigation and request full file from insurer.")
            elif severity == "Medium / 中":
                st.markdown("Advise complainant to contact insurer in writing first. Escalate to IA if no satisfactory reply within 14 days.")
            else:
                st.markdown("Acknowledge positive feedback. No immediate action required.")

            # Store for TTS
            if 'last_analysis' not in st.session_state:
                st.session_state.last_analysis = {}
            st.session_state.last_analysis = {
                "sentiment": sentiment,
                "severity": severity,
                "advice": advice
            }

    # Text-to-Speech
    if speak_btn and 'last_analysis' in st.session_state:
        with st.spinner("Generating audio..."):
            analysis_text = f"Analysis result: {st.session_state.last_analysis['sentiment']}. Severity: {st.session_state.last_analysis['severity']}. Recommendation: {st.session_state.last_analysis['advice']}"
            try:
                speech = tts_pipe(analysis_text[:500])
                st.audio(speech["audio"], sample_rate=speech["sampling_rate"])
                st.success("🔊 Audio ready!")
            except Exception as e:
                st.error(f"Audio failed: {str(e)}")

    # Sidebar
    with st.sidebar:
        st.header("About IA Sentinel")
        st.write("• Sentiment & Severity Analysis")
        st.write("• EasyOCR for document images")
        st.write("• Supports English + Chinese")
        st.info("**Note:** EasyOCR may take time on first load. For best results, use clear document images.")

if __name__ == "__main__":
    main()
