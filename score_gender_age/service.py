from sanic import Sanic
from sanic.response import json
from get_score import main
from get_detail import get_gender_age
import requests
from PIL import Image


app = Sanic(__name__)

@app.route("/get_score", methods=["GET"])
async def get_score(request):
    url = request.args.get("url")
    response = requests.get(url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)

    image = Image.open('image.jpg')
    res = main(image)
    return json(res)


@app.route("/get_detail", methods=["GET"])
async def get_detail(request):
    url = request.args.get("url")
    response = requests.get(url)
    with open('image.jpg', 'wb') as f:
        f.write(response.content)

    image = Image.open('image.jpg')
    res = get_gender_age(image)
    return json(res)


if __name__ == '__main__':
    app.run(host="localhost", port=8090)