import json
import requests
import numpy as np

def show(idx, title):
  plt.figure()
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})

f = np.load('../model/data/mnist.npz')
x_test, y_test = f['x_test'], f['y_test']
x_test = np.reshape(x_test, [-1, 784])

headers = {"content-type": "application/json"}
data = json.dumps({"signature_name": "serving_default", "instances": x_test[0:1].tolist()})

json_response = requests.post('http://localhost:8501/v1/models/saved_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

print(np.argmax(predictions[0]))