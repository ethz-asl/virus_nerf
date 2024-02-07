// # Connection:
// #       Vcc (Arduino)    -> Pin 6 VCC 
// #       GND (Arduino)    -> Pin 7 GND 
// #       Pin 3 (Arduino)  -> Pin 2 ECHO 



int PIN_PWM = 3;         // PWM Output pin
int PIN_NOISE1 = 5;
int PIN_NOISE2 = 6;
unsigned long DELAY = 50000; // delay in us
bool NOISE = false;

void setup()
{
  //Serial initialization
  Serial.begin(9600);                        // Sets the baud rate to 9600

  pinMode(PIN_PWM, INPUT);

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

  unsigned long dist = pulseIn(PIN_PWM, HIGH) ; // every 1us low level stands for 1mm
  Serial.print("meas=");
  Serial.println(dist);

  // unsigned long ellapse_time = micros() - start_time;
  // if(DELAY > ellapse_time) {
  //   delayMicroseconds(DELAY - ellapse_time);
  // }
  delay(10);
  
}
