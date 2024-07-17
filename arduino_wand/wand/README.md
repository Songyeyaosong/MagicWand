Tensorflow转换成TensorflowLite的模型文件是.tflite后缀，需要转成.h才能写进ESP32



在linux或者windows的git bash运行这段代码：

```xxd -i model.tflite > model.h```