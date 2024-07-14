#include <TensorFlowLite_ESP32.h>
#include "model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Wire.h>
#include <MPU6050.h>

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;

const int num_classes = 3;

constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

MPU6050 mpu;

float pitch;
float roll;

float gravity_x;
float gravity_y;
float gravity_z;

// 互补滤波器系数
const float alpha = 0.95;

// 上次更新时间
unsigned long prevTime;

//定义HZ
const int HZ = 64;
const int second = 1;

const int buttonPin = 4; // 定义按钮引脚
const int ledPin = 12;
int buttonState;          // 当前按钮状态
int lastButtonState = HIGH; // 上一次按钮状态

unsigned long lastDebounceTime = 0; // 上一次去抖动时间
unsigned long debounceDelay = 10;   // 去抖动延时

void setup() {
  Serial.begin(115200);

  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddTranspose();
  resolver.AddExpandDims();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  model_input = interpreter->input(0);

  Wire.begin();
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050连接失败");
    while (1);
  }

  pinMode(buttonPin, INPUT_PULLUP); // 将按钮引脚设置为输入模式，并启用内部上拉电阻
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  calibratePR();
}

void loop() {

  int reading = digitalRead(buttonPin); // 读取按钮引脚的电平状态

  // 检查是否有按钮状态变化
  if (reading != lastButtonState) {
    lastDebounceTime = millis(); // 记录状态变化的时间
  }

  // 如果状态变化超过去抖动延时，认为是有效变化
  if ((millis() - lastDebounceTime) > debounceDelay) {
    // 如果按钮状态确实变化了
    if (reading != buttonState) {
      buttonState = reading;

      // 只有在按钮从按下变为释放时，才改变LED状态
      if (buttonState == HIGH) {
        calibratePR();

        for (int i = 0; i < HZ * second; i ++) {
          writeData(i, model_input->data.f);
        }

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed");
          return;
        }

        processGesture(interpreter->output(0)->data.f);
      }
    }
  }

  // 记录上一次按钮状态
  lastButtonState = reading;
}

void calibratePR() {
  // 读取加速度计数据
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  // 转换加速度计数据为g值
  float Ax = ax / 16384.0;
  float Ay = ay / 16384.0;
  float Az = az / 16384.0;

  // 计算Pitch和Roll
  pitch = -atan2(Ax, sqrt(Ay * Ay + Az * Az));
  roll = atan2(Ay, Az);

  prevTime = millis();
}

void writeData(int i, float* input) {
  // 获取当前时间
  unsigned long currentTime = millis();
  float dt = (currentTime - prevTime) / 1000.0; // 时间间隔（秒）
  prevTime = currentTime;

  // 读取加速度计和陀螺仪数据
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // 转换加速度计数据为g值
  float Ax = ax / 16384.0;
  float Ay = ay / 16384.0;
  float Az = az / 16384.0;

  // 转换陀螺仪数据为弧度/秒
  float Gx = gx / 131.0 / 180 * PI;
  float Gy = gy / 131.0 / 180 * PI;
  float Gz = gz / 131.0 / 180 * PI;

  // 使用加速度计数据计算Pitch和Roll
  float accelPitch = -atan2(Ax, sqrt(Ay * Ay + Az * Az));
  float accelRoll = atan2(Ay, Az);

  // 使用陀螺仪数据更新Pitch和Roll
  pitch = alpha * (pitch + Gx * dt) + (1 - alpha) * accelPitch;
  roll = alpha * (roll + Gy * dt) + (1 - alpha) * accelRoll;

  gravity_x = -sin(pitch);
  gravity_y = sin(roll) * cos(pitch);
  gravity_z = cos(roll) * cos(pitch);

  Ax = Ax - gravity_x;
  Ay = Ay - gravity_y;
  Az = Az - gravity_z;

  input[i * 6] = Ax;
  input[i * 6 + 1] = Ay;
  input[i * 6 + 2] = Az;
  input[i * 6 + 3] = Gx;
  input[i * 6 + 4] = Gy;
  input[i * 6 + 5] = Gz;

  delay(1000 / HZ); // 短暂延迟，避免过高的循环频率
}

void processGesture(float* output) {
  int max_index = 0;
  float max_value = output[0];

  for (int i = 1; i < num_classes; i++) {
    if (output[i] >= max_value) {
      max_value = output[i];
      max_index = i;
    }
  }

  if (max_index == 0) {
    digitalWrite(ledPin, LOW);
  } else if (max_index == 1) {
    digitalWrite(ledPin, HIGH);
    delay(500);
    digitalWrite(ledPin, LOW);
  } else {
    digitalWrite(ledPin, HIGH);
    delay(250);
    digitalWrite(ledPin, LOW);
    delay(250);
    digitalWrite(ledPin, HIGH);
    delay(250);
    digitalWrite(ledPin, LOW);
  }
}
