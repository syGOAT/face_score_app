from sanic import Sanic
from sanic.response import json
from video_face_detect import get_faces
import base64

app = Sanic(__name__)

@app.route("/get_details", methods=["GET"])
async def get_details(request):
    url = request.args.get("url")
    res = get_faces(url)
    res = [base64.b64encode(face).decode('utf-8') for face in res]
    return json(res)


if __name__ == '__main__':
    app.run(host="localhost", port=8080)