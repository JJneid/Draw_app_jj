from PIL import Image
import base64
from io import BytesIO

import dash
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash_canvas import DashCanvas
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_daq as daq
from dash_canvas.utils import (array_to_data_url, parse_jsonstring,)

###########
import pandas as pd
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions


app = dash.Dash(__name__)
server = app.server

##################
cats=pd.read_csv("assets/sorted_cats.csv")
cats.columns
cats= cats["sorted_cats"]
################################a

########## defining the model with pre trained weights##############
model=tf.keras.applications.MobileNet(weights='assets/model.h5',input_shape=(64, 64, 1), classes=340)

############# function to read drawing and give prediction#############
def get_prediction(model, img_target):
    img = image.load_img(img_target, target_size=(64, 64),color_mode="grayscale")
    x= image.img_to_array(img)
    x.resize(64, 64)
    result = np.zeros((1, 64, 64, 1))
    result[0,:x.shape[0],:x.shape[1],0] = x
    y = -preprocess_input(result)
    return sorted(zip(model.predict(y)[0], cats), reverse=True)[:3]    


canvas_width = 500
canvas_height = 300

app.layout = html.Div(
    [
        # Banner
        html.Div(
            [
                #html.Img(src=app.get_asset_url("ocr-logo.png"), className="app__logo"),
                html.H4("Make me guess your drawing!", className="header__text"),
            ],
            className="app__header",
        ),
        # Canvas
        html.Div([
            html.Div([
                       html.Div(
                                [
                                    html.H4(
                                        "Draw inside the canvas with your stylus and press Submit",
                                        className="section_title",
                                    ),
                                    #html.H6("You can change the color and width", className="section_title"),
                                    html.Div(
                                    [    DashCanvas(
                                            id="canvas",
                                            lineWidth=8,
                                            width=canvas_width,
                                            height=canvas_height,
                                            #filename=filename,
                                            hide_buttons=[
                                                "zoom",
                                                "pan",
                                            #     "line",
                                            #     "pencil",
                                            #     "rectangle",
                                            #     "select",
                                             ],
                                            add_only=False,
                                            lineColor="black",
                                            goButtonTitle="Submit",
                                        ),
                                     html.Div(
                                        html.Button(id="clear", children="clear drawing"),
                                        className="v-card-content-markdown-outer",
                                    ),                                       
                                        ],
                                        className="canvas-outer",
                                        style={"margin-top": "1em"},
                                    ),
                            html.H6(children=['Brush width']),
                            dcc.Slider(
                                id='bg-width-slider',
                                min=8,
                                max=40,
                                step=1,
                                value=8
                            ),
                                ],
                                className="v-card-content",
                            ),
                            # html.Div(
                            #     html.Button(id="clear1", children="clear drawing"),
                            #     className="v-card-content-markdown-outer",
                            # ),
                    ],className="seven columns"),
                    
                
             html.Div([
                       html.Div(
                                [
                                    html.H4("Humm, it looks like you are drawing: ", className="section_title"),
                                    dcc.Loading(dash_table.DataTable(id="text-output", columns=[{"name": "Possible Categories ?", "id": "Category"}],data=[],style_cell={'textAlign': 'center'},)),
                                ],
                                className="section_title",
                                style={"margin-top": "1em"},
                            ),                ],className="five columns"),                    
                    
                    

               
               ], className="row flex-display"),
               
               
               
                
             html.Div([

                            html.Div([
                                html.H4("Choose one of these categories :) ", className="section_title"),
                                html.Section(id="slideshow", children=[
                                    html.Div(id="slideshow-container", children=[
                                        html.Div(id="image"),
                                        dcc.Interval(id='interval', interval=3000)
                                    ])
                                ])
                            
                            ], className="seven columns"), 
                           html.Div([
                       
                                   html.Div([
                               
                                           html.H4("Your drawing",className="section_title"),
                                           html.Img(id='segmentation-iimg', width=200)],
                                        className="v-card-content",
                                        style={"margin-top": "1em"},),                      
        
                                   ],className="five columns"),
                ],className="row flex-display"),

                
                
            ],
            className="eleven columns",
        )


@app.callback(Output("canvas", "json_objects"), [Input("clear", "n_clicks")])
def clear_canvas(n):
    if n is None:
        return dash.no_update
    strings = ['{"objects":[ ]}', '{"objects":[]}']
    return strings[n % 2]


@app.callback(
    Output("text-output", "data"), [Input('segmentation-iimg', 'src')],
)
def update_data(string):
    if string:
        result=get_prediction(model,"temp/image.png") 
    else:
        raise PreventUpdate
        
    return pd.DataFrame(result, columns = ['Probability','Category']).to_dict('rows')




@app.callback(Output('segmentation-iimg', 'src'),
              [Input('canvas', 'json_data')])
def segmentation(string):
    if string:

        try:
            mask = parse_jsonstring(string, shape=(canvas_height, canvas_width))
        except:
            return "Out of Bounding Box, click clear button and try again"
        # np.savetxt('data.csv', mask) use this to save the canvas annotations as a numpy array
        # Invert True and False
        mask = (~mask.astype(bool)).astype(int)

        image_string=array_to_data_url((255 * mask).astype(np.uint8))

        # this is from canvas.utils.image_string_to_PILImage(image_string)
        img = Image.open(BytesIO(base64.b64decode(image_string[22:])))
        
        #img_rgb= cv2.cvtColor(np.float32(image_string),cv2.COLOR_GRAY2RGB)
        img.save(os.path.join("temp/", 'image.png'), quality=200)
        return img
    

@app.callback(Output('image', 'children'),
              [Input('interval', 'n_intervals')])
def display_image(n):
    if n == None or n % 6 == 1:
        img = html.Img(src=app.get_asset_url('cats_pic1.png'),width=500,height=300)
    elif n % 6 == 2:
        img = html.Img(src=app.get_asset_url('cats_pic2.png'),width=500,height=300)
    elif n % 6 == 3:
        img = html.Img(src=app.get_asset_url('cats_pic3.png'),width=500,height=300)
    elif n % 6 == 4:
        img = html.Img(src=app.get_asset_url('cats_pic4.png'),width=500,height=300)
    elif n % 6 == 5:
        img = html.Img(src=app.get_asset_url('cats_pic5.png'),width=500,height=300)
    elif n % 3 == 0:
        img = html.Img(src=app.get_asset_url('cats_pic6.png'),width=500,height=300)
    else:
        img = "None"
    return img

@app.callback(Output('canvas', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

if __name__ == "__main__":
    app.run_server(debug=True)
