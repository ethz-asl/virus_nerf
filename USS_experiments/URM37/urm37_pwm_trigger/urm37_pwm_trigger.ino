// # Editor     : roker
// # Date       : 05.03.2018

// # Product name: URM V5.0 ultrasonic sensor
// # Product SKU : SEN0001
// # Version     : 1.0

// # Description:
// # The Sketch for scanning 180 degree area 2-800cm detecting range
// # The sketch for using the URM37 PWM trigger pin mode from DFRobot
// #   and writes the values to the serialport
// # Connection:
// #       Vcc (Arduino)    -> Pin 1 VCC (URM V5.0)
// #       GND (Arduino)    -> Pin 2 GND (URM V5.0)
// #       Pin 3 (Arduino)  -> Pin 4 ECHO (URM V5.0)
// #       Pin 5 (Arduino)  -> Pin 6 COMP/TRIG (URM V5.0)

// # Working Mode: PWM trigger pin  mode.

int PIN_ECHO = 3;         // PWM Output 0-50000US,Every 50US represent 1cm
int PIN_TRIG = 4;         // trigger pin
int PIN_NOISE1 = 5;
int PIN_NOISE2 = 6;
unsigned long DELAY = 200000; // delay in us
bool NOISE = false;

void setup()
{
  //Serial initialization
  Serial.begin(9600);                        // Sets the baud rate to 9600

  pinMode(PIN_TRIG, OUTPUT);                   // A low pull on pin COMP/TRIG
  digitalWrite(PIN_TRIG, HIGH);                // Set to HIGH
  pinMode(PIN_ECHO, INPUT);                    // Sending Enable PWM mode command

  pinMode(PIN_NOISE1, OUTPUT);
  pinMode(PIN_NOISE2, OUTPUT);
  digitalWrite(PIN_NOISE1, LOW);
  digitalWrite(PIN_NOISE2, LOW);

  delay(500);
  Serial.println("Init the sensor");

}
void loop()
{
  if(NOISE) {
    digitalWrite(PIN_NOISE1, HIGH);
    digitalWrite(PIN_NOISE2, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_NOISE1, LOW);
    digitalWrite(PIN_NOISE2, LOW);
    delayMicroseconds(2900);
  }

  digitalWrite(PIN_TRIG, LOW);
  delay(5); // short delay that sensor detects trigger signal
  digitalWrite(PIN_TRIG, HIGH);              

  unsigned long pulse = pulseIn(PIN_ECHO, LOW) ;
  unsigned int dist = round(float(pulse) / 50);  // every 50us low level stands for 1cm
  Serial.print("meas=");
  Serial.println(dist);

  // unsigned long ellapse_time = micros() - start_time;
  // Serial.print("Sleep for us:");
  // Serial.println(DELAY - ellapse_time);
  // if(DELAY > ellapse_time) {
  //   delayMicroseconds(DELAY - ellapse_time);
  // }
  delay(10);
}
