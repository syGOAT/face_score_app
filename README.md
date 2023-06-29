# 更新



**requirements**

torch                1.11.0 
numpy                1.23.1 
opencv-python        4.6.0.66 
sanic                23.3.0  



# recognize_boxes

仅一个路由

请求：get，图片url包含在请求url中

响应：人脸坐标集，元素为 float 类型

```json
[
	[ , , , ], 
	...
]
```



# score_gender_age

### 路由1

请求：get，图片url包含在请求url中（图片为切割后的人脸图片）

响应：得分 float

### 路由2

请求：get，图片url包含在请求url中（图片为切割后的人脸图片）

响应：年龄与性别

```json
{
	'gender': str, 
	'age': str
}
```

