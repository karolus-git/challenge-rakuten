import streamlit as st
import streamlit.components.v1 as components
import re
import json
import plotly.express as px
import requests
import pandas as pd
import numpy as np

import components
import utils

#TODO link to .env file
URL_FASTAPI_SERVING_FUSION_P = "http://fastapi:8000/api/v1/predict/fusion"
URL_FASTAPI_SERVING_TEXT_P = "http://fastapi:8000/api/v1/predict/text"
URL_FASTAPI_SERVING_TEXT_E = "http://fastapi:8000/api/v1/explain/text"
URL_FASTAPI_SERVING_IMAGE_P = "http://fastapi:8000/api/v1/predict/image"
URL_FASTAPI_SERVING_IMAGE_E = "http://fastapi:8000/api/v1/explain/image"

#TODO move it somewehere else or handle it as an object ?
label2categorie = {
    10	:	"Livres",
    40	:	"Gaming",
    50	:	"Gaming",
    60	:	"Gaming",
    1140	:	"Jouets",
    1160	:	"Jouets",
    1180	:	"Jouets",
    1280	:	"Jouets",
    1281	:	"Jouets",
    1300	:	"Jouets",
    1301	:	"Bazar",
    1302	:	"Jouets",
    1320	:	"Equipement",
    1560	:	"Mobilier",
    1920	:	"Décoration",
    1940	:	"Bazar",
    2060	:	"Décoration",
    2220	:	"Equipement",
    2280	:	"Livres",
    2403	:	"Livres",
    2462	:	"Gaming",
    2522	:	"Livres",
    2582	:	"Mobilier",
    2583	:	"Equipement",
    2585	:	"Equipement",
    2705	:	"Livres",
    2905	:	"Gaming",
}

prdcodetype2label = {
        10 : "Livre occasion",
        40 : "Jeu vidéo, accessoire tech.",
        50 : "Accessoire Console",
        60 : "Console de jeu",
        1140 : "Figurine",
        1160 : "Carte Collection",
        1180 : "Jeu Plateau",
        1280 : "Jouet enfant, déguisement",
        1281 : "Jeu de société",
        1300 : "Jouet tech",
        1301 : "Paire de chaussettes",
        1302 : "Jeu extérieur, vêtement",
        1320 : "Autour du bébé",
        1560 : "Mobilier intérieur",
        1920 : "Chambre",
        1940 : "Cuisine",
        2060 : "Décoration intérieure",
        2220 : "Animal",
        2280 : "Revues et journaux",
        2403 : "Magazines, livres et BDs",
        2462 : "Jeu occasion",
        2522 : "Bureautique et papeterie",
        2582 : "Mobilier extérieur",
        2583 : "Autour de la piscine",
        2585 : "Bricolage",
        2705 : "Livre neuf",
        2905 : "Jeu PC",
    }


st.set_page_config(
    layout="wide",
    page_title="Rakuten Challenge",
)

#States
if 'text_input' not in st.session_state:
    st.session_state['text_input'] = ''
if 'image_url_input' not in st.session_state:
    st.session_state['image_url_input'] = ''
if 'scrap_input' not in st.session_state:
    st.session_state['scrap_input'] = ''
if 'label' not in st.session_state:
    st.session_state['label'] = "Classification prediction : I will try to guess this :sunglasses:"
if 'confidence' not in st.session_state:
    st.session_state['confidence'] = "Confidence rate : best i can..."

#Run the app
def run():

    #Initialisation
    json_response_p = {}
    json_response_e = {}

    with st.sidebar:

        components.header()

        text_input = components.text_input()
        image_url_input = components.image_url_input()
        scrapper_input = components.scrapper_input()
            
        #Add SHAP parameters
        add_shap = st.checkbox('Add explanations with SHAP? (Time consuming...)') 
        evals_shap = st.slider('How may evaluatins?', 0, 10000, value=100, step=100) 
        #*topk_shap = st.slider('How many top?', 1, 27, 5) 

        #Buttons
        c_sub, c_cle = st.columns(2)

        with c_sub: #Submit inputs to make a prediction
            submit_button = st.button(label='Predict 📈')
            
        with c_cle: #Clean the inputs   
            clear = st.button(label='Clean 🗑️', on_click=utils.clean_forms) 
            
    
    if submit_button:
        with st.spinner('Classifying, please wait....'):
            if not text_input and not image_url_input :
                st.info('It would be easier with some inputs ... ', icon="ℹ️")
                                
            # Text only
            elif text_input and not image_url_input:
                    #Keep only alphanumeric
                    text_for_request = re.sub(r'[^A-Za-z0-9 ]+', '', text_input)
                        
                    #Prepare data for sending text
                    data_instances = {
                        "text" : text_for_request,
                        "evals" : evals_shap,
                        "topk" : 5 #TODO change to variable
                    }                                                                                                                                                                                                           
                    post_url_e = URL_FASTAPI_SERVING_TEXT_E
                    post_url_p = URL_FASTAPI_SERVING_TEXT_P

            #Image and Text
            elif text_input and image_url_input:
                    #Keep only alphanumeric
                    text_for_request = re.sub(r'[^A-Za-z0-9 ]+', '', text_input)
                        
                    #Prepare data for sending text
                    data_instances = {
                        "text" : text_for_request,
                        "image_url" : image_url_input,
                        "evals" : evals_shap,
                        "topk" : 5 #TODO change to variable
                    }                                                                                                                                                                                                           
                    post_url_e = None
                    post_url_p = URL_FASTAPI_SERVING_FUSION_P

            #Image only
            elif not text_input and image_url_input:
                    #Prepare data for sending an image
                    data_instances = {
                        "image_url" : image_url_input,
                        "evals" : evals_shap,
                        "topk" : 5 #TODO change to variable
                    }
                    post_url_e = URL_FASTAPI_SERVING_IMAGE_E
                    post_url_p = URL_FASTAPI_SERVING_IMAGE_P         

            #Post to backend for prediction
            response_p = requests.post(
                    url=post_url_p, 
                    json=data_instances,
                    headers={"content-type": "application/json"})

            #Post to backend for explanation
            if add_shap:
                response_e = requests.post(
                        url=post_url_e, 
                        json=data_instances,
                        headers={"content-type": "application/json"}) 
                json_response_e = json.loads(response_e.text)
                print(json_response_e)

            #Response as json
            json_response_p = json.loads(response_p.text)
                
        # except Exception as exce:
        #     st.error("API could not be reached")  

        #Two main columns (only one if add_shape not checked)
        output_colums_dist = (20, 2, 15) if add_shap else (20,)
        output_colums = st.columns(output_colums_dist)

        
        #If we have a response with predictions
        if (json_response_p and not json_response_p.get("error")):
            
            #TODO Make this better : use saved labelencoder ? get those informations from fastapi?
            results = np.array(json_response_p["results"])
            idx_max = results.argmax()

            print(results)
            value_max = results[idx_max]
            label_max = list(prdcodetype2label.values())[idx_max]

            if value_max > .75:
                color = "green"
            elif .25 <= value_max < .75:
                color = "orange"
            else:
                color = "red"

            st.session_state.label = f"Classification prediction : **{label_max}**"
            st.session_state.confidence = f"Confidence rate : **:{color}[{value_max*100:02.2f}%]**"
            st.write(st.session_state.label)
            st.write(st.session_state.confidence)

            with output_colums[0]:
                                    
                #Add infos to dict
                json_response_p["labels"] = prdcodetype2label.values()
                json_response_p["categories"] = label2categorie.values()
                json_response_p.pop('error')
                
                #Convert the probas to a dataframe
                df = pd.DataFrame.from_dict(json_response_p).reset_index()
                
                #Plot it with plotly
                fig = px.bar(df, x='labels', y='results', color="categories", barmode='group')
                for data in fig.data:
                    data["width"] = 0.55 #Change this value for bar widths
                    
                st.plotly_chart(fig, use_container_width=False)
                
                # #Detailed json
                # with st.expander("Detailed Predictions & Probabilities", expanded=False):
                #     st.write(json_response_p)
        else:
            st.warning("error with the predictions")

        #If we have a response with explanations
        if add_shap:
            if (json_response_e and not json_response_e.get("error")):
                with output_colums[-1]:

                    #Get the shape plot as html
                    shape_values = json_response_e.get("results")

                    #Add it in a html component       
                    st.components.v1.html(shape_values, 
                            width=None, 
                            height=520, 
                            scrolling=True)

                    # #Detailed json
                    # with st.expander("Detailed Explanations", expanded=False):
                    #     st.write(json_response_e)
            else:
                st.warning("error with the explanations")

if __name__ == "__main__":
    run()