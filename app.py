import gradio as gr
from transformers import pipeline
import os

# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to handle analysis
def analyze_text(file, prompt):
    if file is None:
        return "Please upload a .txt file.", "", None

    # Read uploaded .txt file
    with open(file.name, "r", encoding="utf-8") as f:
        content = f.read()

    # Combine prompt and file content
    combined_input = prompt + "\n\n" + content

    # Summarize
    summary = summarizer(combined_input, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

    # Word count
    word_count = len(summary.split())

    # Save summary to a downloadable file
    output_file = "summary_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Summary:\n{summary}\n\nWord Count: {word_count}")

    return summary, f"Word Count: {word_count}", output_file


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI-Powered TXT File Analyzer")
    gr.Markdown("Upload a `.txt` file and enter a prompt (e.g., 'Summarize about AI') to get a smart summary.")

    file_input = gr.File(file_types=[".txt"], label="📄 Upload your .txt file")
    prompt_input = gr.Textbox(placeholder="Enter your prompt here...", label="💬 Prompt (e.g., 'Summarize about AI')")
    
    analyze_btn = gr.Button("Analyze")
    clear_btn = gr.Button("Clear")

    summary_output = gr.Textbox(label="📝 Summary Result")
    word_count_output = gr.Textbox(label="🔢 Word Count")
    download_btn = gr.File(label="⬇️ Download Result")

    analyze_btn.click(analyze_text, inputs=[file_input, prompt_input], outputs=[summary_output, word_count_output, download_btn])
    clear_btn.click(lambda: ("", "", None), outputs=[summary_output, word_count_output, download_btn])

# Launch the app
demo.launch()
