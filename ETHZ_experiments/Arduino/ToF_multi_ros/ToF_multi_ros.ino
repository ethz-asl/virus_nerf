/*
  Read an 8x8 array of distances from the VL53L5CX
  By: Nathan Seidle
  SparkFun Electronics
  Date: October 26, 2021
  License: MIT. See license file for more information but you can
  basically do whatever you want with this code.

  This example shows how to read all 64 distance readings at once.

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/18642

*/
#include <ros.h>
#include <ros/time.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt32.h>
#include <std_msgs/Int16MultiArray.h>
#include <sensors/TOF.h>
#include <Wire.h>

#include <SparkFun_VL53L5CX_Library.h> //http://librarymanager/All#SparkFun_VL53L5CX

#define TOF_I2C_ENABLE_PIN_1  11
#define TOF_I2C_ENABLE_PIN_2  12
#define TOF_I2C_ADDRESS1      0x11 // Valid: 0x08 <= address <= 0x77
#define TOF_I2C_ADDRESS2      0x22 // Valid: 0x08 <= address <= 0x77
#define TOF_INTERUPT_PIN_1    3 //Connect VL53L5CX INT pin to pin 4 on your microcontroller
#define TOF_INTERUPT_PIN_2    4

char FRAME_ID_1[5] = "TOF1";
char FRAME_ID_2[5] = "TOF2";

// std_msgs::Int16MultiArray tof_msg;
sensors::TOF tof_msg_1;
// sensors::TOF tof_msg_2;
ros::NodeHandle nh;
ros::Publisher tof_pub1(FRAME_ID_1, &tof_msg_1);
// ros::Publisher tof_pub2(FRAME_ID_2, &tof_msg_2);

SparkFun_VL53L5CX myImager1;
// SparkFun_VL53L5CX myImager2;
VL53L5CX_ResultsData measurementData1;
// VL53L5CX_ResultsData measurementData2; // Result data class structure, 1356 byes of RAM

int img_res = 0; //Used to pretty print output
int img_width = 0; //Used to pretty print output
bool tof_interrupt_flag_1 = false;
// bool tof_interrupt_flag_2 = false;

void tof_interrupt1() {
  tof_interrupt_flag_1 = true;
}

// void tof_interrupt2() {
//   tof_interrupt_flag_2 = true;
// }

void tof_enableAllI2C(bool enable) {
  if(enable) { // enable I2C of all sensors
    digitalWrite(TOF_I2C_ENABLE_PIN_1, HIGH);
    digitalWrite(TOF_I2C_ENABLE_PIN_2, HIGH);
  } else { // disable I2C of all sensors
    digitalWrite(TOF_I2C_ENABLE_PIN_1, LOW);
    digitalWrite(TOF_I2C_ENABLE_PIN_2, LOW);
  }
}

void tof_startMeas(SparkFun_VL53L5CX* imager) {
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (imager->begin() == false)
  {
    Serial.println(F("Sensor not found - check your wiring. Freezing"));
    while (1) ;
  }
}

void tof_changeAddress(SparkFun_VL53L5CX* imager, int i2c_enable_pin, int i2c_address) {
  // disable I2C of all sensors
  tof_enableAllI2C(false);

  // enable I2C of sensor to change address
  digitalWrite(i2c_enable_pin, HIGH);

  Serial.print(F("Setting sensor ? address to: 0x"));
  Serial.println(i2c_address, HEX);

  if (imager->setAddress(i2c_address) == false)
  {
    Serial.println(F("Sensor ? failed to set new address. Please try again. Freezing..."));
    while (1);
  }
  delay(500);

  int newAddress = imager->getAddress();
  Serial.print(F("New address of sensor 1 is: 0x"));
  Serial.println(newAddress, HEX);

  // enable I2C of all sensors
  tof_enableAllI2C(true);
  delay(500);
}

void tof_publishMeas1() {
  tof_msg_1.header.stamp = nh.now();
  tof_msg_1.header.frame_id = FRAME_ID_1;
  for(int i=0; i<img_res; i++) {
    tof_msg_1.meas[i] = measurementData1.distance_mm[i];
  }
  tof_pub1.publish( &tof_msg_1 );
}

// void tof_publishMeas2() {
//   tof_msg_2.header.stamp = nh.now();
//   tof_msg_2.header.frame_id = FRAME_ID_2;
//   for(int i=0; i<img_res; i++) {
//     tof_msg_2.meas[i] = measurementData2.distance_mm[i];
//   }
//   tof_pub2.publish( &tof_msg_2 );
// }

void tof_printMeas1() {
  //The ST library returns the data transposed from zone mapping shown in datasheet
  //Pretty-print data with increasing y, decreasing x to reflect reality
  for (int y = 0 ; y <= img_width * (img_width - 1) ; y += img_width)
  {
    for (int x = img_width - 1 ; x >= 0 ; x--)
    {
      Serial.print("\t");
      Serial.print("1:");
      Serial.print(measurementData1.distance_mm[x + y]);
    }
    Serial.println();
  }
  Serial.println();
}

// void tof_printMeas2() {
//   //The ST library returns the data transposed from zone mapping shown in datasheet
//   //Pretty-print data with increasing y, decreasing x to reflect reality
//   for (int y = 0 ; y <= img_width * (img_width - 1) ; y += img_width)
//   {
//     for (int x = img_width - 1 ; x >= 0 ; x--)
//     {
//       Serial.print("\t");
//       Serial.print("2:");
//       Serial.print(measurementData2.distance_mm[x + y]);
//     }
//     Serial.println();
//   }
//   Serial.println();
// }

void setup()
{
  // set baud rate for arduino and ros node
  Serial.begin(115200);
  nh.getHardware()->setBaud(115200);
  delay(1000);

  // initialize node
  nh.initNode();
  nh.advertise(tof_pub1);
  // nh.advertise(tof_pub2);

  

  Serial.println("SparkFun VL53L5CX Imager Example");

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //Sensor has max I2C freq of 400kHz 
  
  pinMode(TOF_I2C_ENABLE_PIN_1, OUTPUT);
  // pinMode(TOF_I2C_ENABLE_PIN_2, OUTPUT);
  digitalWrite(TOF_I2C_ENABLE_PIN_1, HIGH);
  // digitalWrite(TOF_I2C_ENABLE_PIN_2, HIGH); 

  tof_startMeas(&myImager1);
  // tof_changeAddress(&myImager1, TOF_I2C_ENABLE_PIN_1, TOF_I2C_ADDRESS1);
  // tof_changeAddress(&myImager2, TOF_I2C_ENABLE_PIN_2, TOF_I2C_ADDRESS2);


  
  //Configure both sensors the same just to keep things clean
  myImager1.setResolution(8 * 8); //Enable all 64 pads
  // myImager2.setResolution(8 * 8); //Enable all 64 pads
  
  img_res = myImager1.getResolution(); //Query sensor for current resolution - either 4x4 or 8x8
  img_width = sqrt(img_res); //Calculate printing width

  myImager1.setRangingFrequency(1);
  // myImager2.setRangingFrequency(1);

  tof_msg_1.meas = (uint16_t*)malloc(sizeof(uint16_t) * img_res);
  tof_msg_1.meas_length = img_res;
  // tof_msg_2.meas = (uint16_t*)malloc(sizeof(uint16_t) * img_res);
  // tof_msg_2.meas_length = img_res;

  // Attach the interrupt
  attachInterrupt(digitalPinToInterrupt(TOF_INTERUPT_PIN_1), tof_interrupt1, FALLING);
  // attachInterrupt(digitalPinToInterrupt(TOF_INTERUPT_PIN_2), tof_interrupt2, FALLING);
  Serial.println(F("Interrupt pins configured."));

  myImager1.startRanging();
  // myImager2.startRanging();
  Serial.println("Start ranging");
}

void loop()
{
  if(tof_interrupt_flag_1) {
    tof_interrupt_flag_1 = false;

    if (myImager1.getRangingData(&measurementData1)) { //Read distance data into array
      tof_publishMeas1();
      tof_printMeas1();
      
    }
  }
  // if(tof_interrupt_flag_2) {
  //   tof_interrupt_flag_2 = false;

  //   if (myImager2.getRangingData(&measurementData2)) { //Read distance data into array
  //     tof_publishMeas2();
  //     tof_printMeas2();
  //   }
  // }

  nh.spinOnce();
  delay(5); //Small delay between polling
}
