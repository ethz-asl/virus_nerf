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
// #       Pin 4 (Arduino)  -> Pin 6 COMP/TRIG (URM V5.0)

// # Working Mode: PWM trigger pin  mode.

#include <ros.h>
#include <ros/time.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt32.h>
#include <sensors/USS.h>



int PIN_ECHO = 3;         // PWM Output 0-50000US,Every 50US represent 1cm
int PIN_TRIG = 4;         // trigger pin
unsigned long DELAY = 1000; // delay in ms
char FRAME_ID[5] = "USS0";

//std_msgs::UInt32 uint32_msg;
sensors::USS uss_msg;
ros::NodeHandle nh;
ros::Publisher urm37(FRAME_ID, &uss_msg);

void setup()
{
  // set baud rate for arduino and ros node
  Serial.begin(57600);
  nh.getHardware()->setBaud(57600);

  // initialize node
  nh.initNode();
  nh.advertise(urm37);

  // define and set IOs
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_TRIG, OUTPUT);
  digitalWrite(PIN_TRIG, HIGH);

  delay(500);

}
void loop()
{
  // trigger measurement
  digitalWrite(PIN_TRIG, LOW);
  delay(5); // short delay that sensor detects trigger signal
  digitalWrite(PIN_TRIG, HIGH);              

  // make measurement
  uint32_t pulse = pulseIn(PIN_ECHO, LOW);

  // publish measurement
  // uint32_msg.data = pulse;
  uss_msg.meas = pulse;
  uss_msg.header.frame_id = FRAME_ID;
  uss_msg.header.stamp = nh.now();
  urm37.publish( &uss_msg );

  // unsigned long pulse = pulseIn(PIN_ECHO, LOW) ;
  // Serial.println(pulse);

  nh.spinOnce();
  delay(DELAY);
}
