import streamlit as st
import scrapper

#Clean the forms
def clean_forms():
    st.session_state["text_input"] = ""
    st.session_state["image_url_input"] = ""
    st.session_state["scrap_input"] = ""

#Fill text_input and image_url_input with scrapper
def fill_scrap():
    scrapped = scrapper.scrap(st.session_state.scrap_input)
    st.session_state["text_input"] = scrapped.get("text_input")
    st.session_state["image_url_input"] = scrapped.get("image_url_input")
