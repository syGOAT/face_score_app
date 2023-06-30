# 更新



**requirements**

torch                1.11.0 
numpy                1.23.1 
opencv-python        4.6.0.66 
sanic                23.3.0  



# recognize_boxes

请求：get，图片url包含在请求url中

响应：人脸坐标集，元素为 float 类型

```json
[
	[ , , , ], 
	...
]
```



# score_detail

请求：get，图片url包含在请求url中（图片为切割后的人脸图片）

响应：

```json
{
    'score': float,
	'gender': str, 
	'age': str, 
    'idol': {
        'zsw.jpg': float, 
        ...
    }
}
```

在 idol 字典中，每个键对应的值为“距离损失函数值”，**越小越匹配**



| 文件名  | 明星姓名      |
| ------- | ------------- |
| zsw.jpg | 张颂文        |
| zjy.jpg | 张婧仪        |
| zrf.jpg | 周润发        |
| lhq.jpg | 刘华强        |
| xlz.jpg | 莱昂纳多      |
| mm.jpg  | 泰勒·斯威夫特 |

