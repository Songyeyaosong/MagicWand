ESP32+MPU6050+TensorFlow的一个魔杖项目



Tensorflow转换成TensorflowLite的模型文件是.tflite后缀，需要转成.h才能写进ESP32



在linux系统（或者windows的git bash）运行这段代码将.tflite转成.h：

```xxd -i model.tflite > model.h```