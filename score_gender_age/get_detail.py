import cv2



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


def get_gender_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)   # blob输入网络进行性别的检测
    genderPreds = genderNet.forward()   # 性别检测进行前向传播
    gender = genderList[genderPreds[0].argmax()]   # 分类  返回性别类型
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    return {
        'gender': gender, 
        'age': age
    }
    