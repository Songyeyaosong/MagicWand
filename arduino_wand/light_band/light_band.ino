#include <Adafruit_NeoPixel.h>

#define NUM_LEDS 144
#define DATA_PIN 14

Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_LEDS, DATA_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.show();
}

void loop() {
  energyGatherAndBurst(20, 1); // 调整这些值来控制效果速度
}

void energyGatherAndBurst(uint8_t gatherDelay, uint8_t burstDelay) {
  int baseLeds = 3; // 底部亮起的LED数量

  // 聚集效果 - 底部LED逐渐变亮
  for (int brightness = 0; brightness <= 255; brightness += 5) {
    for (int i = 0; i < baseLeds; i++) {
      strip.setPixelColor(i, strip.Color(brightness, brightness, brightness));
    }
    strip.show();
    delay(gatherDelay);
  }

  // 爆发效果 - LED从底部快速移动到顶部
  strip.clear();
  for (int i = 0; i < baseLeds; i++) {
    strip.setPixelColor(i, strip.Color(255, 255, 255));
  }
  strip.show();
  delay(burstDelay);

  for (int i = 1; i <= strip.numPixels() - baseLeds; i++) {
    strip.clear();
    for (int j = 0; j < baseLeds; j++) {
      strip.setPixelColor(i + j, strip.Color(255, 255, 255));
    }
    strip.show();
    delay(burstDelay);
  }

  // 清除顶部亮光
  strip.clear();
  strip.show();
  delay(500); // 增加一个小的延迟以区分效果循环
}
