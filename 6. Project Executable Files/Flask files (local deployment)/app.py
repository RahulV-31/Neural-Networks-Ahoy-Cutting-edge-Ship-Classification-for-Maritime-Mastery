import numpy as np
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from keras.layers import Flatten
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

from flask import Flask , request, render_template
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
basepath = os.path.dirname(__file__)
modelpath = os.path.join(basepath,'uploads',"vgg16-ship-classification.h5")
model = load_model(modelpath)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        
        #print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(224,224))
        img=image.img_to_array(img)
        img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        img=preprocess_input(img)
        pred=model.predict(img)
        pred=pred.flatten()
        pred=list(pred)
        n=max(pred)
        val_dict={0: 'Aircraft Carrier',
        1: 'Bulkers',
        2: 'Car Carrier',
        3: 'Container Ship',
        4: 'Cruise',
        5: 'DDG',
        6: 'Recreational',
        7: 'Sailboat',
        8: 'Submarine',
        9: 'Tug'}
        result=val_dict[pred.index(n)]
        print(result)
        text = "the Ship Category is " + result
    return text

if __name__ == '__main__':
    app.run(debug = False)
        
        