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
    response = requests.get(url)
    ranint = random.randint(0, 99999)
    with open('image_{}.jpg'.format(ranint), 'wb') as f:
        f.write(response.content)

    image = Image.open('image_{}.jpg'.format(ranint))
    res1 = main(image)
    res2 = get_gender_age(image)
    return json(res1.update(res2))



if __name__ == '__main__':
    app.run(host="localhost", port=8090)