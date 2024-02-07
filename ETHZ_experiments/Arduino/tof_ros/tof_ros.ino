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

#define INT_PIN 4 //Connect VL53L5CX INT pin to pin 4 on your microcontroller
char FRAME_ID[5] = "TOF0";

// std_msgs::Int16MultiArray tof_msg;
sensors::TOF tof_msg;
ros::NodeHandle nh;
ros::Publisher tof_pub(FRAME_ID, &tof_msg);

SparkFun_VL53L5CX myImager;
VL53L5CX_ResultsData measurementData; // Result data class structure, 1356 byes of RAM

int img_res = 0; //Used to pretty print output
int img_width = 0; //Used to pretty print output
bool interrupt_flag = false;

void interruptRoutine() {
  interrupt_flag = true;
}

void tof_publishMeas() {
  tof_msg.header.stamp = nh.now();
  tof_msg.header.frame_id = FRAME_ID;
  for(int i=0; i<img_res; i++) {
    tof_msg.meas[i] = measurementData.distance_mm[i];
  }
  tof_pub.publish( &tof_msg );
}

void tof_printMeas() {
  //The ST library returns the data transposed from zone mapping shown in datasheet
  //Pretty-print data with increasing y, decreasing x to reflect reality
  for (int y = 0 ; y <= img_width * (img_width - 1) ; y += img_width)
  {
    for (int x = img_width - 1 ; x >= 0 ; x--)
    {
      Serial.print("\t");
      Serial.print(measurementData.distance_mm[x + y]);
    }
    Serial.println();
  }
  Serial.println();
}

void setup()
{
  // set baud rate for arduino and ros node
  Serial.begin(115200);
  nh.getHardware()->setBaud(115200);
  delay(1000);

  // initialize node
  nh.initNode();
  nh.advertise(tof_pub);

  

  Serial.println("SparkFun VL53L5CX Imager Example");

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //Sensor has max I2C freq of 400kHz 
  
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (myImager.begin() == false)
  {
    Serial.println(F("Sensor not found - check your wiring. Freezing"));
    while (1) ;
  }
  
  myImager.setResolution(8*8); //Enable all 64 pads
  
  img_res = myImager.getResolution(); //Query sensor for current resolution - either 4x4 or 8x8
  img_width = sqrt(img_res); //Calculate printing width

  tof_msg.meas = (uint16_t*)malloc(sizeof(uint16_t) * img_res);
  tof_msg.meas_length = img_res;

  // Attach the interrupt
  attachInterrupt(digitalPinToInterrupt(INT_PIN), interruptRoutine, FALLING);
  Serial.println(F("Interrupt pin configured."));

  myImager.startRanging();
}

void loop()
{
  if(interrupt_flag) {
    interrupt_flag = false;

    if (myImager.getRangingData(&measurementData)) { //Read distance data into array
      tof_publishMeas();
      //tof_printMeas();
    }
  }

  nh.spinOnce();
  delay(5); //Small delay between polling
}
