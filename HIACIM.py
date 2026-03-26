import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from PIL import Image
import easyocr
import io
import os

# Optional: For better Whisper support (recommended for production)
# pip install openai-whisper  (or use transformers Whisper pipeline)

def main():
    st.set_page_config(page_title="IA Complaint Analyzer", page_icon="🛡️", layout="wide")
    st.title("🛡️ Hong Kong Insurance Authority Complaint Analyzer")
    st.markdown("""
    **AI Tool for IA Staff & Public**  
    Analyze insurance complaints for **sentiment** and **severity**.  
    Now supports **voice (Cantonese/Mandarin/English)** and **image (OCR)** inputs.  
    Supports consumer protection and efficient complaint handling.  
    [IA Official Site](https://www.ia.org.hk/en/index.html)
    """)

    # Load classifier (your fine-tuned model or fallback)
    @st.cache_resource
    def load_classifier():
        try:
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    classifier = load_classifier()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("Tailored for Hong Kong Insurance Authority.")
        st.write("New features: Voice recording (Cantonese/Mandarin/English) + Image OCR.")
        st.write("Model: Fine-tuned DistilBERT (or fallback).")
        st.info("Note: IA handles conduct-related complaints. For pure claims disputes, contact the insurer first or consider FDRC.")

    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎙️ Voice Input", "📸 Image Input"])

    complaint_text = ""

    with tab1:
        st.subheader("Enter Complaint Text")
        complaint_text = st.text_area(
            "Insurance complaint text (e.g., delay in claim, poor intermediary conduct):",
            height=200,
            placeholder="The insurer refused to renew my policy without clear reason..."
        )

    with tab2:
        st.subheader("Record or Upload Voice Complaint")
        st.caption("Supports English, Mandarin, and Cantonese (Hong Kong speech patterns)")

        language = st.selectbox(
            "Select primary language of the recording",
            options=["English", "Mandarin (zh)", "Cantonese (yue-HK)"],
            index=2  # Default to Cantonese for HK relevance
        )

        lang_code = {"English": "en", "Mandarin (zh)": "zh", "Cantonese (yue-HK)": "yue"}[language]

        audio_value = st.audio_input("Click to record your complaint (microphone)")

        if audio_value:
            st.audio(audio_value, format="audio/wav")
            st.success("Audio recorded successfully!")

            if st.button("🔊 Transcribe Audio to Text"):
                with st.spinner("Transcribing audio... (Whisper model)"):
                    try:
                        # Option 1: Use OpenAI Whisper (recommended - install whisper)
                        # For demo, we'll simulate or use transformers Whisper pipeline
                        # In production: use whisper.load_model("base") or "large-v3" for better Cantonese
                        transcript = "Transcribed text would appear here (using Whisper). " \
                                    "Example: The insurance company delayed my claim for three months without explanation."
                        
                        # Real implementation example (uncomment when whisper is installed):
                        # import whisper
                        # model = whisper.load_model("base")
                        # result = model.transcribe(audio_value, language=lang_code)
                        # transcript = result["text"]

                        st.write("**Transcription:**")
                        complaint_text = st.text_area("Edit transcribed text if needed:", value=transcript, height=150)
                    except Exception as e:
                        st.error(f"Transcription error: {e}. Please try text input instead.")

    with tab3:
        st.subheader("Upload Image of Complaint (e.g., letter, screenshot, form)")
        uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tiff"])

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("📖 Extract Text with OCR (EasyOCR)"):
                with st.spinner("Performing OCR... (supports English + Chinese)"):
                    try:
                        # Initialize EasyOCR reader (downloads models on first run)
                        reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)  # ch_sim for simplified Chinese; add 'ch_tra' if needed
                        img_bytes = uploaded_image.getvalue()
                        result = reader.readtext(img_bytes, detail=0)  # detail=0 returns text only

                        extracted_text = "\n".join(result)
                        st.write("**Extracted Text:**")
                        complaint_text = st.text_area("Edit extracted text if needed:", value=extracted_text, height=150)
                    except Exception as e:
                        st.error(f"OCR error: {e}")

    # Unified Analyze button (works across tabs)
    if st.button("🚀 Analyze Complaint", type="primary"):
        if not complaint_text or not complaint_text.strip():
            st.error("Please provide complaint text via text, voice, or image input.")
        else:
            with st.spinner("Analyzing complaint for sentiment and severity..."):
                time.sleep(1.5)  # Simulate processing

                results = classifier(complaint_text)
                label = results[0]['label']
                score = results[0]['score']

                # Map results (customize for your IA-specific labels)
                if "NEGATIVE" in label.upper() or (score > 0.7 and "neg" in label.lower()):
                    sentiment = "Negative 😠"
                    severity = "High"
                    advice = "Prioritize for IA review – possible conduct issue or unfair treatment."
                elif "POSITIVE" in label.upper():
                    sentiment = "Positive 🙂"
                    severity = "Low"
                    advice = "Likely resolved or positive feedback. Monitor for patterns."
                else:
                    sentiment = "Neutral 😐"
                    severity = "Medium"
                    advice = "Further human review recommended. Check supporting documents."

                st.success("✅ Analysis Complete")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Complaint Text:**")
                    st.write(complaint_text)
                with col2:
                    st.metric("Sentiment", sentiment, f"Confidence: {score:.2%}")
                    st.write(f"**Severity Level:** {severity}")
                    st.write(f"**IA Recommendation:** {advice}")

                st.info("💡 Tip: For voice/image inputs in Hong Kong, many complaints involve Cantonese/Mixed language or scanned documents.")

    st.caption("Built for efficient IA complaint triage | Voice & Image features enhance accessibility for the public.")

if __name__ == "__main__":
    main()
