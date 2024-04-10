import io
import json
import random
from flask import jsonify

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch


# 启动服务时就直接加载好模型
static_dir='/static'

app = Flask(__name__, static_url_path= static_dir)

@app.route('/model_generate/<promtMsg>')
def get_stable_diffusion_v1_4(promtMsg):
    print(promtMsg)
    image = pipe(promtMsg.strip()).images[0]
    
    # 随机生成文件命字，存在static file中，并返回静态url让前端调用
    tmp_img_path= static_dir + "/"+ str(hash(promtMsg))+".jpg"
    image.save("."+ tmp_img_path)
    print(tmp_img_path)
    return jsonify(
        imgUrl=tmp_img_path
    )


if __name__ == '__main__':
    app.run()