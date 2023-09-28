
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from  tensorflow import keras
import pickle
import streamlit as st

#import streamlit as st
#import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
#import cv2
from PIL import Image, ImageOps
import numpy as np

model1 = keras.models.load_model('model_1.h5')

model2 = keras.models.load_model('model_1.h5')

model3 = keras.models.load_model('model_VGG.h5')

diseases = ['Potato___Hollow_Heart',
 'Squash___Powdery_mildew',
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Tomato___Late_blight',
 'Strawberry___Leaf_scorch',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Early_blight',
 'Tomato___Tomato_mosaic_virus',
 'Potato___Late_blight',
 'Tomato___healthy',
 'Grape___healthy',
 'Grape___Black_rot',
 'Pepper,_bell___healthy',
 'Tomato___Canker',
 'Corn_(maize)___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Peach___healthy',
 'Soybean___healthy',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Apple___Rotten',
 'Corn_(maize)___Common_rust_',
 'Tomato___Septoria_leaf_spot',
 'Grape___Esca_(Black_Measles)',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Tea__Black_rot',
 'Potato___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Peach___Bacterial_spot',
 'Raspberry___healthy',
 'Blueberry___healthy',
 'Tea__Healthy',
 'Tomato___Leaf_Mold',
 'Tomato___Bacterial_spot',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Ginger__Healthy',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Target_Spot',
 'Strawberry___healthy',
 'Potato___Early_blight']



st.write("""
         # Plant Disease Detection
         """
         )
file = st.file_uploader("Upload the image to be classified ", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model):
    
        """
        size = (256,256)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
        """
        
        prediction = model.predict(upload_image)
        index=prediction.argmax(axis=-1)[0]
        class_name = diseases[index]
        
        return class_name

if file is None:
    st.text("Please upload an image file")
else:
    #image1 = Image.open(file)
    new_img =keras.utils.load_img(file, target_size=(256, 256))
    img = keras.utils.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255


    st.image(file, use_column_width=True)
    predictions = upload_predict(img, model2)
    #image_class = str(predictions[0][0][1])
    #score=np.round(predictions[0][0][2]) 
    st.write("The image is classified as",predictions)
    #st.write("The similarity score is approximately",score)
    print("The image is classified as ",predictions)#, "with a similarity score of",score)
