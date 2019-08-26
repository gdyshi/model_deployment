from flask import Flask, request
import json
import numpy as np
import sys
import traceback
from python_model import python_model

app = Flask(__name__)

model = python_model(model_path='../model/saved_pb/tensorflow.pb')


@app.route('/inference', methods=['POST'])
def inference():
    result = {}

    try:
        file = request.files['image']
        file.save('tmp_image.dat')
        x_test = np.load('tmp_image.dat')
        output = model.inference(x_test)
        print(output.astype(np.int32))

        result['ret'] = 0
        result['msg'] = 'success'
        result['result'] = output.to_list
    except Exception as e:
        print('{} error {}'.format(sys._getframe().f_code.co_name, traceback.format_exc()))
        result['ret'] = 0
        result['msg'] = e.args[0]
    finally:
        print(result)
        return json.dumps(result, ensure_ascii=False, default=lambda o: o.__dict__)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)
