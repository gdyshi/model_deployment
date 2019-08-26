import os
import requests

http_url = 'http://127.0.0.1:5003'

def inference(file_path):
    files = {}

    if not os.path.exists(file_path):
        return None
    files['image'] = (os.path.basename(file_path), open(file_path, 'rb'))
    response = requests.post(http_url+'/inference', files=files)
    result = response.json()
    result['httpcode'] = response.status_code

    if 'result' in result:
        return result['result']
    else:
        return None


if __name__ == '__main__':
    print(inference(r'data.npy'))

