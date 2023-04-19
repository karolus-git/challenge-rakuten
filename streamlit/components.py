import streamlit as st
import utils

def header():
    return st.header("The Rakuten Challenge, powered by PyTorch, FastAPI, Shap, Docker & Hydra", anchor=None)

#Text input
def text_input():
    return st.text_area("Text", 
        height=100,
        key="text_input", 
        placeholder="Fill this form to predict text (in french for better performances!)")

#Image URL input
def image_url_input():
    return st.text_input("Image URL", 
        key="image_url_input", 
        placeholder="Fill this form with an url pointing to an image to predict image")

#Scrapper URL input
def scrapper_input():
    return st.text_input(
        "Scrap URL", 
        key="scrap_input", 
        placeholder="URL from Rakuten or RueDuCommercer + Enter",
        on_change=utils.fill_scrap)
            
