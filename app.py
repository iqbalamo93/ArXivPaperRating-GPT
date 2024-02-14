import streamlit as st
import pandas as pd
from datetime import datetime
import glob
import os


def load_data():
    # Assuming there's only one CSV file in the 'data' directory
    file_path = glob.glob('./data/*.csv')[0]  # Gets the first matching CSV file
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    return df, file_path

def load_markdown():
    # Load the Markdown text from the file
    with open('./data/markdown_text.md', 'r') as file:
        markdown_text = file.read()

    # Optionally remove bold formatting if present
    # This line of code replaces '**text**' or '__text__' with 'text'
    markdown_text = markdown_text.replace('**', '').replace('__', '')


    return markdown_text

def main():
    # App layout and styling
    st.set_page_config(page_title="📊 Research Paper Rating Analysis", layout="wide")

    # Header section
    st.title("📚 Research Paper Rating Analysis")
    st.subheader("🌟 Project: Rating Research Papers in the Last 7 Days Using AI")
    
    # Load the data and extract the file path
    df, file_path = load_data()

    # Extract the last run date from the filename
    file_date_str = os.path.basename(file_path).split('.')[0]  # Extracts the date part from the filename
    file_date = datetime.strptime(file_date_str, 'data_%d_%m_%Y').date()
    st.write(f"This project was last run on: {file_date.strftime('%B %d, %Y')}")  # Formatting date as Month day, Year

    # Disclaimer
    st.info("ℹ️ Disclaimer: This analysis is a study to gauge potential biases in academic judgments by machine learning models and should not be considered as conclusive evidence of the quality or relevance of the research papers.")

    # Display the Markdown text - Center aligned with white color
    st.markdown("<h2 style='text-align: center; color: white;'>📝 Top Rated Research Papers</h2>", unsafe_allow_html=True)
    markdown_text = load_markdown()
    st.markdown(markdown_text, unsafe_allow_html=True)

    # Display the DataFrame
    st.markdown("### 📊 Detailed Ratings Table")
    st.dataframe(df)

    # Footer
    st.markdown("---")
    st.markdown("© 2024 LLM Based Rating Project. All rights reserved.")

if __name__ == "__main__":
    main()