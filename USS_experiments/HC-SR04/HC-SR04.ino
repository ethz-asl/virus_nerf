/*
  Ultrasonic Sensor HC-SR04 and Arduino Tutorial

  by Dejan Nedelkovski,
  www.HowToMechatronics.com

*/


// defines pins numbers
const int PIN_ECHO = 3;         
const int PIN_TRIG = 5;         // trigger pin
const int PIN_NOISE1 = 50;
const int PIN_NOISE2 = 51;

const unsigned long DELAY = 200000; // delay in us
const bool NOISE = false;



void setup() {
  Serial.begin(9600); // Starts the serial communication

  pinMode(PIN_ECHO, INPUT); // Sets the echoPin as an Input
  pinMode(PIN_TRIG, OUTPUT); // Sets the trigPin as an Output
  digitalWrite(PIN_TRIG, LOW);
  
  // pinMode(PIN_NOISE1, OUTPUT);
  // pinMode(PIN_NOISE2, OUTPUT);
  // digitalWrite(PIN_NOISE1, LOW);
  // digitalWrite(PIN_NOISE2, LOW);

  delay(500);
  Serial.println("Init the sensor");
}


void loop() {
  // if(NOISE) {
  //   digitalWrite(PIN_NOISE1, HIGH);
  //   digitalWrite(PIN_NOISE2, HIGH);
  //   delayMicroseconds(10);
  //   digitalWrite(PIN_NOISE1, LOW);
  //   digitalWrite(PIN_NOISE2, LOW);
  //   delayMicroseconds(2900);
  // }

  // Sets the trigPin on HIGH state for 10 micro seconds
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  // Reads the PIN_ECHO, returns the sound wave travel time in microseconds
  unsigned long pulse = pulseIn(PIN_ECHO, HIGH);
  unsigned int dist = round(float(pulse) * 0.34 / 2);
  Serial.print("meas=");
  Serial.println(dist);

  delay(10);
}
