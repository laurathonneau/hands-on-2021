import base64
import io
import yaml

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from PIL import Image

from constants import CLASSES

#load parameters from the yaml file

with open('app.yaml') as yaml_data:
    
    parameters = yaml.safe_load(yaml_data)


IMAGE_WIDTH = parameters[0]['IMAGE_WIDTH']
IMAGE_HEIGHT = parameters[1]['IMAGE_HEIGHT']
MODEL_PATH = parameters[2]['MODEL_PATH']



# Load DNN model
classifier = tf.keras.models.load_model(MODEL_PATH)

def classify_image(image, model, image_box=None):
  """Classify image by model

  Parameters
  ----------
  content: image content
  model: tf/keras classifier

  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return [np.argmax(model.predict(np.array(images_list)))]


def classify_image_proba(image, model, image_box=None):
  """Classify image by model

  Parameters
  ----------
  path: filepath to image
  model: tf/keras classifier

  Returns
  -------
  probability of the class returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return [np.amax(model.predict(np.array(images_list)))]


def classify_image_probavec(image, model, image_box=None):
  """Classify image by model

  Parameters
  ----------
  path: filepath to image
  model: tf/keras classifier

  Returns
  -------
  array of probabilities of all classes returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return model.predict(np.array(images_list))



app = dash.Dash('Traffic Signs Recognition') #, external_stylesheets=dbc.themes.BOOTSTRAP)

colors = {
    'background': '#e5f2e5',
    'text': '#000066'
}

pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}



# Define application layout
app.layout = html.Div(style={'backgroundColor':colors['background']}, children=[
    html.H1(
        children='Reconnaissance de Panneaux de Signalisation',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(
        children='Cette application permet de reconnaitre des panneaux de signalisation, présents sur des photos, grace à des un modèle de réseaux de neurones profonds.', 
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(
        children='Télécharger ci-dessous une photo de panneau à reconnaitre : ', 
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    dcc.Upload(
            id='bouton-chargement',
            children=html.Div([
                'Cliquer-déposer ou ',
                        html.A('sélectionner une image')
            ]),
            style={
                'color': colors['text'],
                'width': '50%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',                
                'backgroundColor': colors['background'],
                'margin' : '40px',
                'marginLeft': '480px',
                'maginBottom': '40px'

            }
    ),
    html.Div(id='mon-image')
])


    

@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            associated_proba = classify_image_proba(image, classifier)[0]
            probs = classify_image_probavec(image, classifier)[0]
            df = pd.DataFrame({
                "Probabilité": np.reshape(probs, (43,)),
                "Catégorie": CLASSES.values()
            })
            fig = px.bar(df, x="Catégorie", y="Probabilité", color= CLASSES.items())
            fig.update_layout(title_font_color= colors['text'],
                              title_font_size = 18 ,
                              title={
                                  'text': "Probabilité estimée et associée à chaque type de Panneau",
                                  'y':0.9, 
                                  'x':0.5,
                                  'xanchor':'center',
                                  'yanchor':'top'})            
            return html.Div([
                html.Hr(),
                html.Img(src=contents),
                html.H3('Le panneau de circulation prédit est {!r} avec une probabilité associée de {:.2%}'.format(CLASSES[predicted_class], associated_proba)),  
                html.Hr(),
                dcc.Graph(figure=fig), 

                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                associated_proba = classify_image_proba(image, classifier)[0]
                probs = classify_image_probavec(image, classifier)[0]
                df = pd.DataFrame({
                    "Probabilité": np.reshape(probs, (43,)),
                    "Catégorie": CLASSES.values()
                })
                fig = px.bar(df, x="Catégorie", y="Probabilité", color= CLASSES.items(), title='Probabilité associée à chaque type de Panneau')
                fig.update_layout(title_font_color= colors['text'],
                                  title_font_size = 18 ,
                                  title={
                                      'text': "Probabilité estimée et associée à chaque type de Panneau",
                                      'y':0.9, 
                                      'x':0.5,
                                      'xanchor':'center',
                                      'yanchor':'top'})                
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string),
                    html.H3('Le panneau de circulation prédit est {!r} avec une probabilité associée de {:.2%}'.format(CLASSES[predicted_class],associated_proba)),
                    html.Hr(),
                    dcc.Graph(figure=fig), 
                    
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
            

# Manage interactions with callbacks
"""
@app.callback(
    Output(component_id='ma-zone-resultat', component_property='children'),
    [Input(component_id='mon-champ-texte', component_property='value')]
)
def update_output_div(input_value):
    return html.H3('Valeur saisie ici "{}"'.format(input_value))
"""

# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)
