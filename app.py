from flask import Flask,render_template,request
from keras.models import model_from_json
from keras.preprocessing import image
import base64
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

app=Flask(__name__,template_folder="templates")

with open('model/model.json','r') as f:
	load_json=f.read()
model=model_from_json(load_json)
model.load_weights('model/model.h5')#weights in model.h5 file

@app.route('/')
def dos():
	return render_template("index.html")

@app.route('/data',methods=['POST','GET'])
def fumx():
	if request.method=="POST":
		data=str(request.form.get('image'))
		base_Data=data.encode('ascii')
		data=base64.b64decode(base_Data)
		with open('imageToSave.png','wb') as fh:
			fh.write(data)
		img=image.load_img('imageToSave.png',target_size=(28,28),grayscale=True)
		res=np.argmax(model.predict(np.expand_dims(img,0)))
		return render_template("index.html",ans=str(res))
	else:
		return "NULL"
	
if __name__ == '__main__':
	app.run(debug=True)
