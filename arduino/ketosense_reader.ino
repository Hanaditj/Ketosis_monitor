/*
 * KetosenseX - Breath Acetone Detector
 * Author: Hanadi Thaisir Jaradath
 * Hardware: Arduino Uno + MQ-3 Gas Sensor + DHT11
 *
 * Reads breath acetone (proxy for blood ketone levels),
 * compensates for temperature/humidity, and sends
 * structured JSON to Serial for Python pipeline.
 */

#include <DHT.h>
#include <ArduinoJson.h>

#define MQ3_PIN     A0
#define DHT_PIN     7
#define DHT_TYPE    DHT11
#define BAUD_RATE   9600

// Calibration constants (tune after baseline readings)
#define RL_VALUE        5.0    // Load resistance in kOhm
#define RO_CLEAN_AIR    9.83   // Sensor resistance in clean air
#define ACETONE_CURVE_A 0.53   // Log-log slope for acetone
#define ACETONE_CURVE_B -0.36  // Log-log intercept

// Ketosis threshold in ppm (breath acetone)
// <2 ppm  = no ketosis
// 2-40 ppm = nutritional ketosis
// >40 ppm  = deep/starvation ketosis (caution)
#define KETOSIS_LOW_PPM   2.0
#define KETOSIS_HIGH_PPM  40.0

DHT dht(DHT_PIN, DHT_TYPE);

float Ro = RO_CLEAN_AIR;  // Will be recalibrated on startup

// --------------- Sensor Maths ---------------

float readMQ3Voltage() {
  int raw = analogRead(MQ3_PIN);
  return (raw / 1023.0) * 5.0;
}

float computeRS(float voltage) {
  if (voltage < 0.01) voltage = 0.01;  // Avoid divide-by-zero
  return RL_VALUE * (5.0 - voltage) / voltage;
}

float computePPM(float rs) {
  // Log-log linear model: log(ppm) = A * log(RS/Ro) + B
  float ratio = rs / Ro;
  if (ratio <= 0) ratio = 0.001;
  return pow(10, (ACETONE_CURVE_A * log10(ratio) + ACETONE_CURVE_B));
}

// Temperature-humidity compensation factor
float tempHumidityCompensation(float temp, float humidity) {
  // Empirical correction for MQ-series sensors
  // Based on Figaro application notes
  float tempFactor = 1.0 + 0.01 * (temp - 20.0);
  float humFactor  = 1.0 - 0.005 * (humidity - 65.0);
  return tempFactor * humFactor;
}

String classifyKetosis(float ppm) {
  if (ppm < KETOSIS_LOW_PPM)  return "none";
  if (ppm < 10.0)             return "light";
  if (ppm < KETOSIS_HIGH_PPM) return "nutritional";
  return "deep";
}

// --------------- Setup & Loop ---------------

void setup() {
  Serial.begin(BAUD_RATE);
  dht.begin();

  // Warm-up: MQ sensors need ~20s to stabilise
  Serial.println("{\"status\":\"warming_up\",\"duration_s\":20}");
  delay(20000);

  // Calibrate Ro in clean ambient air (average 50 readings)
  float roSum = 0;
  for (int i = 0; i < 50; i++) {
    float v  = readMQ3Voltage();
    roSum   += computeRS(v);
    delay(100);
  }
  Ro = roSum / 50.0;

  Serial.print("{\"status\":\"calibrated\",\"Ro\":");
  Serial.print(Ro, 3);
  Serial.println("}");
}

void loop() {
  float temp     = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temp) || isnan(humidity)) {
    Serial.println("{\"error\":\"dht_read_fail\"}");
    delay(2000);
    return;
  }

  float voltage    = readMQ3Voltage();
  float rs         = computeRS(voltage);
  float correction = tempHumidityCompensation(temp, humidity);
  float rsCorrected = rs * correction;
  float ppm        = computePPM(rsCorrected);
  float mmol       = ppm * 0.0572;  // 1 ppm acetone ≈ 0.0572 mmol/L blood ketone

  String state = classifyKetosis(ppm);

  // Emit structured JSON to Serial — Python reads this
  StaticJsonDocument<256> doc;
  doc["ts"]       = millis();
  doc["temp_c"]   = round(temp * 10) / 10.0;
  doc["humidity"] = round(humidity * 10) / 10.0;
  doc["ppm"]      = round(ppm * 100) / 100.0;
  doc["mmol"]     = round(mmol * 1000) / 1000.0;
  doc["rs"]       = round(rs * 100) / 100.0;
  doc["state"]    = state;
  doc["voltage"]  = round(voltage * 100) / 100.0;

  serializeJson(doc, Serial);
  Serial.println();

  delay(3000);  // 1 reading every 3 seconds
}
