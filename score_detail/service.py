from sanic import Sanic
from sanic.response import json
from get_score import main
from get_detail import get_gender_age
import requests
from PIL import Image
import random


app = Sanic(__name__)

@app.route("/get_score", methods=["GET"])
async def get_score(request):
    url = request.args.get("url")
    face_b = requests.get(url).content
    res1 = main(face_b)
    res2 = get_gender_age(face_b)
    res1.update(res2)
    return json(res1)



if __name__ == '__main__':
    app.run(host="localhost", port=8090)