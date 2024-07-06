const int irSensorPin = 2;
const int drySensorPin = 3;
const int recycleSensorPin = 4;
const int hazardousSensorPin = 5;

void setup() {
  Serial.begin(9600);

  pinMode(irSensorPin, INPUT);
  pinMode(drySensorPin, INPUT);
  pinMode(recycleSensorPin, INPUT);
  pinMode(hazardousSensorPin, INPUT);
}

void loop() {
  int irValue = digitalRead(irSensorPin);
  int dryValue = digitalRead(drySensorPin);
  int recycleValue = digitalRead(recycleSensorPin);
  int hazardousValue = digitalRead(hazardousSensorPin);

  Serial.println("Sensor Readings:");
  Serial.print("WET: ");
  Serial.println(irValue == LOW ? "Full" : "Empty");

  Serial.print("Dry: ");
  Serial.println(dryValue == LOW ? "Full" : "Empty");

  Serial.print("Recycle: ");
  Serial.println(recycleValue == LOW ? "Full" : "Empty");

  Serial.print("Hazardous: ");
  Serial.println(hazardousValue == LOW ? "Full" : "Empty");

  delay(1000);
}
