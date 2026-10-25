#include <Wire.h>

#define MPU6050_ADDR 0x68

#define SDA_PIN 21
#define SCL_PIN 22

// Gyroscope offsets
float gyroXOffset = 0;
float gyroYOffset = 0;
float gyroZOffset = 0;

// Filtered orientation
float angleX = 0;
float angleY = 0;

// Initial reference
float referenceAngle = 0;

// Time
unsigned long previousTime;


// ======================================================
// SETUP
// ======================================================

void setup()
{
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);

  // Wake MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Accelerometer ±2g
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Gyroscope ±250 deg/s
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Serial.println();
  Serial.println("====================================");
  Serial.println("     OA KNEE MOTION MEASUREMENT");
  Serial.println("     SINGLE MPU6050 + ESP32");
  Serial.println("====================================");

  Serial.println();
  Serial.println("Keep your leg STRAIGHT and STILL.");
  Serial.println("Calibration will start in 3 seconds...");

  delay(3000);

  // Gyroscope calibration
  calibrateGyroscope();

  Serial.println();
  Serial.println("Gyroscope calibration completed.");

  // Calculate initial angle
  float initialAngle = getAccelerometerAngle();

  angleY = initialAngle;
  referenceAngle = initialAngle;

  previousTime = millis();

  Serial.println();
  Serial.println("REFERENCE POSITION SET = 0 degrees");
  Serial.println("Now bend and straighten your knee.");
  Serial.println("------------------------------------");
}


// ======================================================
// GYROSCOPE CALIBRATION
// ======================================================

void calibrateGyroscope()
{
  const int samples = 1000;

  float sumX = 0;
  float sumY = 0;
  float sumZ = 0;

  Serial.println();
  Serial.println("Calibrating gyroscope...");
  Serial.println("DO NOT MOVE THE SENSOR!");

  for (int i = 0; i < samples; i++)
  {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    readMPU(ax, ay, az, gx, gy, gz);

    sumX += gx / 131.0;
    sumY += gy / 131.0;
    sumZ += gz / 131.0;

    delay(2);
  }

  gyroXOffset = sumX / samples;
  gyroYOffset = sumY / samples;
  gyroZOffset = sumZ / samples;

  Serial.print("X Offset = ");
  Serial.println(gyroXOffset, 4);

  Serial.print("Y Offset = ");
  Serial.println(gyroYOffset, 4);

  Serial.print("Z Offset = ");
  Serial.println(gyroZOffset, 4);
}


// ======================================================
// READ MPU6050
// ======================================================

void readMPU(
  int16_t &AcX,
  int16_t &AcY,
  int16_t &AcZ,
  int16_t &GyX,
  int16_t &GyY,
  int16_t &GyZ)
{
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU6050_ADDR, 14, true);

  AcX = Wire.read() << 8 | Wire.read();
  AcY = Wire.read() << 8 | Wire.read();
  AcZ = Wire.read() << 8 | Wire.read();

  // Skip temperature
  Wire.read();
  Wire.read();

  GyX = Wire.read() << 8 | Wire.read();
  GyY = Wire.read() << 8 | Wire.read();
  GyZ = Wire.read() << 8 | Wire.read();
}


// ======================================================
// ACCELEROMETER ANGLE
// ======================================================

float getAccelerometerAngle()
{
  int16_t AcX, AcY, AcZ;
  int16_t GyX, GyY, GyZ;

  readMPU(AcX, AcY, AcZ, GyX, GyY, GyZ);

  float ax = AcX / 16384.0;
  float ay = AcY / 16384.0;
  float az = AcZ / 16384.0;

  // Pitch angle
  float angle =
    atan2(-ax, sqrt(ay * ay + az * az))
    * 180.0 / PI;

  return angle;
}


// ======================================================
// MAIN LOOP
// ======================================================

void loop()
{
  int16_t AcX, AcY, AcZ;
  int16_t GyX, GyY, GyZ;

  readMPU(AcX, AcY, AcZ, GyX, GyY, GyZ);

  // ------------------------------
  // ACCELEROMETER
  // ------------------------------

  float ax = AcX / 16384.0;
  float ay = AcY / 16384.0;
  float az = AcZ / 16384.0;


  // ------------------------------
  // GYROSCOPE
  // ------------------------------

  float gx = (GyX / 131.0) - gyroXOffset;
  float gy = (GyY / 131.0) - gyroYOffset;
  float gz = (GyZ / 131.0) - gyroZOffset;


  // ------------------------------
  // ACCELEROMETER ANGLE
  // ------------------------------

  float accelAngleY =
    atan2(-ax, sqrt(ay * ay + az * az))
    * 180.0 / PI;


  // ------------------------------
  // TIME
  // ------------------------------

  unsigned long currentTime = millis();

  float dt =
    (currentTime - previousTime) / 1000.0;

  previousTime = currentTime;


  // ------------------------------
  // GYROSCOPE INTEGRATION
  // ------------------------------

  angleY = angleY + gy * dt;


  // ------------------------------
  // COMPLEMENTARY FILTER
  // ------------------------------

  angleY =
    0.98 * angleY +
    0.02 * accelAngleY;


  // ------------------------------
  // KNEE FLEXION ESTIMATE
  // ------------------------------

  float kneeAngle =
    angleY - referenceAngle;

  // Make angle positive
  if (kneeAngle < 0)
  {
    kneeAngle = -kneeAngle;
  }


  // ------------------------------
  // DISPLAY
  // ------------------------------

  Serial.print("ACC  X: ");
  Serial.print(ax, 2);

  Serial.print("  Y: ");
  Serial.print(ay, 2);

  Serial.print("  Z: ");
  Serial.print(az, 2);

  Serial.println(" g");


  Serial.print("GYRO X: ");
  Serial.print(gx, 2);

  Serial.print("  Y: ");
  Serial.print(gy, 2);

  Serial.print("  Z: ");
  Serial.print(gz, 2);

  Serial.println(" deg/s");


  Serial.print("ORIENTATION: ");
  Serial.print(angleY, 2);

  Serial.println(" deg");


  Serial.print("ESTIMATED KNEE FLEXION: ");
  Serial.print(kneeAngle, 2);

  Serial.println(" deg");


  Serial.println("------------------------------------");

  delay(50);
}