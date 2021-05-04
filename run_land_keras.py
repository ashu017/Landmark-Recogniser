# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = load_model('land_model.h5')
graph = []
labels = {0: 'Ajantha Ellora Caves', 1: 'Basilica of Bom Jesus', 2: 'Gateway of India', 3: 'Gol Gumbaz', 4: 'Hampi', 5: 'Howrah Bridge', 6: 'India Gate', 7: 'Jantar Mantar', 8: 'Konark Temple', 9: 'Lotus Temple', 10: 'Matri Mandir', 11: 'Meenakshi Temple', 12: 'Puri Jagnnath Temple', 13: 'Qutub Minar', 14: 'Red Fort', 15: 'Sanchi Stupa', 16: 'Taj Mahal'}

def loadModel():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	global graph
	model = load_model('land_model.h5')
	# graph = tf.compat.v1.get_default_graph

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = image/255
	# return the processed image
	return image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))


			# classify the input image and then initialize the list
			# of predictions to return to the client
			# with graph.as_default():
				
			preds = model.predict(image)

			i = np.argmax(preds,axis = 1).squeeze()
			i = int(i)
			data["prediction"] = labels[i]

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	loadModel()
	app.run()
