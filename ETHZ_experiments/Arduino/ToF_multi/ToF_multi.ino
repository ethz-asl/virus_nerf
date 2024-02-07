/*
  Read an 8x8 array of distances from the VL53L5CX
  By: Nathan Seidle
  SparkFun Electronics
  Date: October 26, 2021
  License: MIT. See license file for more information but you can
  basically do whatever you want with this code.

  This example shows how to setup and read two sensors. We will hold one
  sensor in reset while we configure the first. You will need to solder
  a wire to each of the sensor's RST pins and connect them to GPIO 14 and 13
  on your plateform.

  Note: The I2C address for the device is stored in NVM so it will have to be set
  at each power on.

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/18642
*/

#include <Wire.h>

#include <SparkFun_VL53L5CX_Library.h> //http://librarymanager/All#SparkFun_VL53L5CX

#define TOF_I2C_ENABLE_PIN_1  11
#define TOF_I2C_ENABLE_PIN_2  12
#define TOF_I2C_ADDRESS1      0x11 // Valid: 0x08 <= address <= 0x77
#define TOF_I2C_ADDRESS2      0x22 // Valid: 0x08 <= address <= 0x77

int imageResolution = 0; //Used to pretty print output
int imageWidth = 0; //Used to pretty print output


SparkFun_VL53L5CX myImager1;
SparkFun_VL53L5CX myImager2;
VL53L5CX_ResultsData measurementData1;
VL53L5CX_ResultsData measurementData2;

void tof_enableAllI2C(bool enable) {
  if(enable) { // enable I2C of all sensors
    digitalWrite(TOF_I2C_ENABLE_PIN_1, HIGH);
    digitalWrite(TOF_I2C_ENABLE_PIN_2, HIGH);
  } else { // disable I2C of all sensors
    digitalWrite(TOF_I2C_ENABLE_PIN_1, LOW);
    digitalWrite(TOF_I2C_ENABLE_PIN_2, LOW);
  }
}

void tof_changeAddress(SparkFun_VL53L5CX* imager, int i2c_enable_pin, int i2c_address) {
  // disable I2C of all sensors
  tof_enableAllI2C(false);

  // enable I2C of sensor to change address
  digitalWrite(i2c_enable_pin, HIGH);

  Serial.println(F("Initializing sensor ?. This can take up to 10s. Please wait."));
  if (imager->begin() == false)
  {
    Serial.println(F("Sensor ? not found. Check wiring. Freezing..."));
    while (1) ;
  }

  Serial.print(F("Setting sensor ? address to: 0x"));
  Serial.println(i2c_address, HEX);

  if (imager->setAddress(i2c_address) == false)
  {
    Serial.println(F("Sensor ? failed to set new address. Please try again. Freezing..."));
    while (1);
  }

  int newAddress = imager->getAddress();
  Serial.print(F("New address of sensor 1 is: 0x"));
  Serial.println(newAddress, HEX);

  // enable I2C of all sensors
  tof_enableAllI2C(true);
}

void setup()
{

  Serial.begin(115200);
  delay(1000);
  Serial.println("SparkFun VL53L5CX Imager Example");

  Wire.begin(); //This resets I2C bus to 100kHz
  Wire.setClock(1000000); //Sensor has max I2C freq of 1MHz

  pinMode(TOF_I2C_ENABLE_PIN_1, OUTPUT);
  pinMode(TOF_I2C_ENABLE_PIN_2, OUTPUT);
  digitalWrite(TOF_I2C_ENABLE_PIN_1, HIGH);
  digitalWrite(TOF_I2C_ENABLE_PIN_2, HIGH); 

  tof_changeAddress(&myImager1, TOF_I2C_ENABLE_PIN_1, TOF_I2C_ADDRESS1);
  tof_changeAddress(&myImager2, TOF_I2C_ENABLE_PIN_2, TOF_I2C_ADDRESS2);



  //Configure both sensors the same just to keep things clean
  myImager1.setResolution(8 * 8); //Enable all 64 pads
  myImager2.setResolution(8 * 8); //Enable all 64 pads

  imageResolution = myImager1.getResolution(); //Query sensor for current resolution - either 4x4 or 8x8
  imageWidth = sqrt(imageResolution); //Calculate printing width

  myImager1.setRangingFrequency(1);
  myImager2.setRangingFrequency(1);

  myImager1.startRanging();
  myImager2.startRanging();
  Serial.println("Start ranging");
}

void loop()
{
  //Poll sensor for new data
  if (myImager1.isDataReady() == true)
  {
    if (myImager1.getRangingData(&measurementData1)) //Read distance data into array
    {
      //The ST library returns the data transposed from zone mapping shown in datasheet
      //Pretty-print data with increasing y, decreasing x to reflect reality
      for (int y = 0 ; y <= imageWidth * (imageWidth - 1) ; y += imageWidth)
      {
        for (int x = imageWidth - 1 ; x >= 0 ; x--)
        {
          Serial.print("\t");
          Serial.print("1:");
          Serial.print(measurementData1.distance_mm[x + y]);
        }
        Serial.println();
      }
      Serial.println();
    }
  }

  if (myImager2.isDataReady() == true)
  {
    if (myImager2.getRangingData(&measurementData2)) //Read distance data into array
    {
      //The ST library returns the data transposed from zone mapping shown in datasheet
      //Pretty-print data with increasing y, decreasing x to reflect reality
      for (int y = 0 ; y <= imageWidth * (imageWidth - 1) ; y += imageWidth)
      {
        for (int x = imageWidth - 1 ; x >= 0 ; x--)
        {
          Serial.print("\t");
          Serial.print("2:");
          Serial.print(measurementData2.distance_mm[x + y]);
        }
        Serial.println();
      }
      Serial.println();
    }
  }

  delay(5); //Small delay between polling
}