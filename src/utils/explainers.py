import shap
import PIL
import torch
import numpy as np
import transformers
import matplotlib.pyplot as plt
import base64

from utils import preprocessors
from utils import predictors

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

prdcodetype2category = {
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

#Encoder
le = torch.load('obj_labelencoder.pth')
prdtypecodes = le.inverse_transform(np.arange(27))
prdtypenames = [f"{prdcodetype2label[code]} | " for code in prdtypecodes]

def image(images, model, topk=5, n_evals=5000, batch_size = 50):
    
    #Transformation tools
    transform = preprocessors.transform_image()
    inv_transform = preprocessors.invert_transform_image()

    def predict_image(image) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(torch.Tensor(image))
            outputs_numpy = outputs.detach().numpy()
        return predictors.softmax(outputs_numpy)

    # Explain your image with topk examples
    images_transformed = torch.stack([transform(image) for image in images])

    # Masker for images
    masker_blur = shap.maskers.Image("blur(128,128)", (224, 224, 3))

    #Get the explainer
    explainer = shap.Explainer(
        predict_image, 
        masker_blur, 
        output_names=prdtypenames)
     
    # Explain the images
    shap_values = explainer(
        images_transformed,
        max_evals=n_evals, 
        batch_size=batch_size,
        outputs=shap.Explanation.argsort.flip[:topk])

    # Tranform back for output
    shap_values.data = shap_values.data[0].numpy().transpose()
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]
    shap_values.values = [val for val in np.moveaxis(shap_values.values, 1, -1)]
    
    #Plot in figure without showing
    html_values = shap.image_plot(
        shap_values=shap_values.values,
        pixel_values=shap_values.data,
        labels=shap_values.output_names,
        show=False)
    
    #Save the plot
    plt.savefig("temp/image_shaped.jpg",dpi=150, bbox_inches='tight')


    #Reload it ... for html
    data_uri = base64.b64encode(open("temp/image_shaped.jpg", 'rb').read()).decode('utf-8')
    html_values = '<img width="400" height="150" src="data:image/png;base64,{0}">'.format(data_uri)
    return html_values

def text(texts, model, topk=5, n_evals=5, batch_size = 50):
        
    def predict_text(texts) -> torch.Tensor:
        return predictors.text(texts, model)

    masker = shap.maskers.Text(r"\W")

    explainer = shap.Explainer(
        predict_text, 
        masker,
        outputs=shap.Explanation.argsort.flip[:topk],
        output_names=prdtypenames)

    shap_values = explainer(
        texts,
        max_evals=n_evals, )

    html_values = shap.plots.text(shap_values, display=False)

    return html_values