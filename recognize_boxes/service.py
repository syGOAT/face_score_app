from sanic import Sanic
from sanic.response import json
from video_face_detect import get_boxes

app = Sanic(__name__)

@app.route("/get_details", methods=["GET"])
async def get_details(request):
    url = request.args.get("url")
    res = get_boxes(url)
    return json(res)


if __name__ == '__main__':
    app.run(host="localhost", port=8080)