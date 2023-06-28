import sys
import cv2

from vision.ssd.config.fd_config import define_img_size


net_type = 'RFB'
input_size = 480
threshold = 0.7
candidate_size = 1000
path = 'imgs'  # img dir
test_device = 'cuda:0'

input_img_size = input_size
define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/voc-model-labels.txt"

net_type = net_type


class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = test_device

model_path = "models/pretrained/version-RFB-320.pth"
# model_path = "models/pretrained/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
net.load(model_path)


def bigger_box(box, type=0):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    w, h = x1 - x0, y1 - y0
    h_new = h * 1.08
    if type == 0:
        x0_new, x1_new = x0 - (h_new - w) / 2, x1 + (h_new - w) / 2
        return [x0_new, y0, x1_new, y0 + h_new]


# 年龄、性别预测
ageProto = "age_gender/age_deploy.prototxt"
ageModel = "age_gender/age_net.caffemodel"
genderProto = "age_gender/gender_deploy.prototxt"
genderModel = "age_gender/gender_net.caffemodel"
# 模型均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
# 加载网络
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


def boxes_details(video_path):
    cap = cv2.VideoCapture(video_path)  # capture from video
    #cap = cv2.VideoCapture(0)  # capture from camera
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            print("end")
            break
        orig_image=cv2.resize(orig_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        faces_details = []
        # boxes里的是乱序，后面按x轴排序一下
        # 暂时默认只有一个人脸，i = 0
        if boxes.size(0) == 0:
            return None
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f" {probs[i]:.2f}"
            new_box = bigger_box(box)
            ###
            #cv2.rectangle(orig_image, (int(new_box[0]), int(new_box[1])), (int(new_box[2]), int(new_box[3])), (0, 255, 0), 4)
            # cv2.putText(orig_image, label,
            #             (box[0], box[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5,  # font scale
            #             (0, 0, 255),
            #             2)  # line type
            face = orig_image[int(new_box[1]): int(new_box[3]), int(new_box[0]): int(new_box[2])]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)   # blob输入网络进行性别的检测
            genderPreds = genderNet.forward()   # 性别检测进行前向传播
            gender = genderList[genderPreds[0].argmax()]   # 分类  返回性别类型
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            new_box = [ts.item() for ts in new_box]
            faces_details.append(
                {'coordinate': new_box, 'gender': gender, 'age': age}
            )
        break
    cap.release()
    return faces_details


if __name__ == '__main__':
    res = boxes_details('http://focnal.xyz:8082/test1.jpg')
    print(res)