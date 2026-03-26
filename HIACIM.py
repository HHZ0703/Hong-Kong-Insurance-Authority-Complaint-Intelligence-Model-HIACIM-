import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

def main():
    st.set_page_config(page_title="IA Complaint Analyzer", page_icon="🛡️", layout="centered")
    st.title("🛡️ Hong Kong Insurance Authority Complaint Analyzer")
    st.markdown("""
    **AI Tool for IA Staff & Public**  
    Analyze insurance complaints for **sentiment** and **severity**.  
    Supports consumer protection and efficient complaint handling.  
    [IA Official Site](https://www.ia.org.hk/en/index.html)
    """)
    
    # Load model (use your fine-tuned or fallback to pipeline)
    @st.cache_resource
    def load_classifier():
        try:
            # Try fine-tuned model (uploaded to HF or local)
            model = AutoModelForSequenceClassification.from_pretrained("IA_Complaint_Classifier")
            tokenizer = AutoTokenizer.from_pretrained("IA_Complaint_Classifier")
            return pipeline("text-classification", model=model, tokenizer=tokenizer)
        except:
            # Fallback to general sentiment (demo)
            return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    classifier = load_classifier()
    
    # User input
    complaint_text = st.text_area(
        "Enter the insurance complaint text (e.g., delay in claim, poor intermediary conduct):",
        height=150,
        placeholder="The insurer refused to renew my policy without clear reason..."
    )
    
    if st.button("Analyze Complaint"):
        if not complaint_text.strip():
            st.error("Please enter complaint text.")
        else:
            with st.spinner("Analyzing... (This supports IA's timely complaint handling)"):
                time.sleep(1)  # Simulate processing
                results = classifier(complaint_text)
                
                # Map results (adapt based on your labels)
                label = results[0]['label']
                score = results[0]['score']
                
                # Simple mapping for demo (Negative → High severity, etc.)
                if "NEGATIVE" in label.upper() or score > 0.7 and "neg" in label.lower():
                    sentiment = "Negative"
                    severity = "High"
                    advice = "Prioritize for IA review – possible conduct issue."
                elif "POSITIVE" in label.upper():
                    sentiment = "Positive"
                    severity = "Low"
                    advice = "Likely resolved or positive feedback."
                else:
                    sentiment = "Neutral"
                    severity = "Medium"
                    advice = "Further human review recommended."
                
                st.success("Analysis Complete")
                st.write("**Complaint Text:**", complaint_text)
                st.write(f"**Sentiment:** {sentiment} (Confidence: {score:.4f})")
                st.write(f"**Severity Level:** {severity}")
                st.write(f"**IA Recommendation:** {advice}")
                
                # Additional IA context
                st.info("Note: IA handles conduct-related complaints. For pure claims disputes, contact the insurer first or consider FDRC.")
    
    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.write("Tailored for Hong Kong Insurance Authority.")
        st.write("Functions: Complaint triage, consumer protection support.")
        st.write("Model: Fine-tuned DistilBERT on HF datasets.")
        if st.button("Download Fine-tuned Model (Colab generated)"):
            st.write("In Colab: Use the zip download feature.")

if __name__ == "__main__":
    main()
