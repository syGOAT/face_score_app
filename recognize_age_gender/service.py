from sanic import Sanic
from sanic.response import json
from video_face_detect import boxes_details

app = Sanic(__name__)

@app.route("/get_details", methods=["GET"])
async def get_details(request):
    url = request.args.get("url")
    res = boxes_details(url)
    return json(res)


if __name__ == '__main__':
    app.run(host="localhost", port=8080)