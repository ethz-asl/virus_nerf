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
// #       Pin 6 (Arduino)  -> Pin 4 ECHO (URM V5.0)
// #       Pin 7 (Arduino)  -> Pin 6 COMP/TRIG (URM V5.0)

// # Working Mode: PWM trigger pin  mode.

#include <Wire.h>
#include <ros.h>
#include <ros/time.h>
#include <sensors/USS.h>
#include <sensors/TOF.h>
#include <SparkFun_VL53L5CX_Library.h> //http://librarymanager/All#SparkFun_VL53L5CX


/*
* ----- DEFINITIONS -----
*/

// General
//#define PRINT_DATA // comment this line to disable printing

// USS
#define USS_TRIGGER_PIN        0
#define USS_INTERRUPT_PIN_1    1
#define USS_INTERRUPT_PIN_2    2
#define USS_INTERRUPT_PIN_3    3
#define USS_ID_1              "USS1"
#define USS_ID_2              "USS2"
#define USS_ID_3              "USS3"

// ToF
#define TOF_IMG_FREQ          15
#define TOF_IMG_RES           64
#define TOF_IMG_WIDTH         8
#define TOF_I2C_RESET_PIN_1   8
#define TOF_I2C_RESET_PIN_2   10
#define TOF_I2C_RESET_PIN_3   12
#define TOF_INTERRUPT_PIN_1   9 //Connect VL53L5CX INT pin to pin 4 on your microcontroller
#define TOF_INTERRUPT_PIN_2   11
#define TOF_INTERRUPT_PIN_3   13
#define TOF_I2C_ADDRESS1      0x11 // Valid: 0x08 <= address <= 0x77
#define TOF_I2C_ADDRESS2      0x22 // Valid: 0x08 <= address <= 0x77
#define TOF_I2C_ADDRESS3      0x33 // Valid: 0x08 <= address <= 0x77
#define TOF_ID_1              "TOF1"
#define TOF_ID_2              "TOF2"
#define TOF_ID_3              "TOF3"

// Testing
#define USS_TRIG_PERIODE_US   500000 // in us, equivalent to 2 Hz


/*
* ----- GLOBAL VARIABLES -----
*/

// ROS
ros::NodeHandle nh;
sensors::USS uss_msg_1;
sensors::USS uss_msg_2;
sensors::USS uss_msg_3;
sensors::TOF tof_msg_1;
sensors::TOF tof_msg_2;
sensors::TOF tof_msg_3;
ros::Publisher uss_pub_1(USS_ID_1, &uss_msg_1);
ros::Publisher uss_pub_2(USS_ID_2, &uss_msg_2);
ros::Publisher uss_pub_3(USS_ID_3, &uss_msg_3);
ros::Publisher tof_pub_1(TOF_ID_1, &tof_msg_1);
ros::Publisher tof_pub_2(TOF_ID_2, &tof_msg_2);
ros::Publisher tof_pub_3(TOF_ID_3, &tof_msg_3);

// USS
volatile bool uss_interrupt_flag_1 = false;
volatile bool uss_interrupt_flag_2 = false;
volatile bool uss_interrupt_flag_3 = false;
volatile uint32_t uss_echo_start_1 = micros();
volatile uint32_t uss_echo_start_2 = micros();
volatile uint32_t uss_echo_start_3 = micros();
volatile uint32_t uss_echo_end_1 = micros();
volatile uint32_t uss_echo_end_2 = micros();
volatile uint32_t uss_echo_end_3 = micros();
ros::Time uss_interrupt_time_1 = nh.now();
ros::Time uss_interrupt_time_2 = nh.now();
ros::Time uss_interrupt_time_3 = nh.now();

// ToF
SparkFun_VL53L5CX tof_imager_1;
SparkFun_VL53L5CX tof_imager_2;
SparkFun_VL53L5CX tof_imager_3;
VL53L5CX_ResultsData tof_data_1; // Result data class structure, 1356 byes of RAM
VL53L5CX_ResultsData tof_data_2; // Result data class structure, 1356 byes of RAM
VL53L5CX_ResultsData tof_data_3; // Result data class structure, 1356 byes of RAM
bool tof_interrupt_flag_1 = false;
bool tof_interrupt_flag_2 = false;
bool tof_interrupt_flag_3 = false;
ros::Time tof_interrupt_time_1 = nh.now();
ros::Time tof_interrupt_time_2 = nh.now();
ros::Time tof_interrupt_time_3 = nh.now();

// Testing
volatile bool uss_new_meas_flag_1 = true;
volatile bool uss_new_meas_flag_2 = true;
volatile bool uss_new_meas_flag_3 = true;
volatile uint32_t uss_trig_time = micros();


/*
* ----- INTERRUPT ROUTINES -----
*/

void uss_interrupt1() {
  if (digitalRead(USS_INTERRUPT_PIN_1) == LOW) {
    uss_echo_start_1 = micros();
  } else {
    uss_echo_end_1 = micros();
    uss_interrupt_time_1 = nh.now();
    uss_interrupt_flag_1 = true;
  }
}

void uss_interrupt2() {
  if (digitalRead(USS_INTERRUPT_PIN_2) == LOW) {
    uss_echo_start_2 = micros();
  } else {
    uss_echo_end_2 = micros();
    uss_interrupt_time_2 = nh.now();
    uss_interrupt_flag_2 = true;
  }
}

void uss_interrupt3() {
  if (digitalRead(USS_INTERRUPT_PIN_3) == LOW) {
    uss_echo_start_3 = micros();
  } else {
    uss_echo_end_3 = micros();
    uss_interrupt_time_3 = nh.now();
    uss_interrupt_flag_3 = true;
  }
}

void tof_interrupt1() {
  tof_interrupt_flag_1 = true;
  tof_interrupt_time_1 = nh.now();
}

void tof_interrupt2() {
  tof_interrupt_flag_2 = true;
  tof_interrupt_time_2 = nh.now();
}

void tof_interrupt3() {
  tof_interrupt_flag_3 = true;
  tof_interrupt_time_3 = nh.now();
}


/*
* ----- FUNCTIONS -----
*/

void tof_initSensor(SparkFun_VL53L5CX *imager, int address, int i2c_reset_pin) {
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");

  // release I2C from reset mode
  digitalWrite(i2c_reset_pin, LOW); 
  
  if (imager->begin() == false)
  {
    Serial.println(F("Sensor not found - check your wiring. Freezing"));
    while (1) ;
  }
  if (imager->setAddress(address) == false)
  {
    Serial.println(F("Sensor ? failed to set new address. Please try again. Freezing..."));
    while (1);
  }
}

void uss_publishMeas(ros::Publisher *pub, sensors::USS *msg, uint32_t echo_duration, 
                      char *id, ros::Time *time) {
  msg->header.stamp = *time;
  msg->header.frame_id = id;
  msg->meas = echo_duration;
  pub->publish( msg );
}

void tof_publishMeas(ros::Publisher *pub, sensors::TOF *msg, VL53L5CX_ResultsData *data, 
                      char *id, ros::Time *time) {
  msg->header.stamp = *time;
  msg->header.frame_id = id;
  for(int i=0; i<TOF_IMG_RES; i++) {
    msg->meas[i] = data->distance_mm[i];
  }
  pub->publish( msg );
}

void uss_printMeas(uint32_t echo_duration, char *id) {
  float distance_cm = (float)echo_duration / 50;
  Serial.print(id);
  Serial.print(": ");
  Serial.println(distance_cm);
}

void tof_printMeas(VL53L5CX_ResultsData *data, char *id) {
  //The ST library returns the data transposed from zone mapping shown in datasheet
  //Pretty-print data with increasing y, decreasing x to reflect reality
  Serial.println(id);
  for (int y = 0 ; y <= TOF_IMG_WIDTH * (TOF_IMG_WIDTH - 1) ; y += TOF_IMG_WIDTH)
  {
    for (int x = TOF_IMG_WIDTH - 1 ; x >= 0 ; x--)
    {
      Serial.print("\t");
      //Serial.print(data->distance_mm[x + y]);
      // Serial.print(data->nb_target_detected[x + y]);
      Serial.print(data->target_status[x + y]);
      // Serial.print(data->target_range_sigma_mm[x + y]);
    }
    Serial.println();
  }
  Serial.println();
} 

// TODO: write status verification function

/*
* ----- SETUP -----
*/

void setup()
{
  // Rosserial communication
  Serial.begin(115200);
  nh.getHardware()->setBaud(115200); // set baud rate for arduino and ros node

  // initialize node
  nh.initNode();
  nh.advertise(uss_pub_1);
  nh.advertise(uss_pub_2);
  nh.advertise(uss_pub_3);
  nh.advertise(tof_pub_1);
  nh.advertise(tof_pub_2);
  nh.advertise(tof_pub_3);

  // USS: IOs
  pinMode(USS_TRIGGER_PIN, OUTPUT);
  digitalWrite(USS_TRIGGER_PIN, HIGH);

  // ToF: IOs 
  pinMode(TOF_I2C_RESET_PIN_1, OUTPUT);
  pinMode(TOF_I2C_RESET_PIN_2, OUTPUT);
  pinMode(TOF_I2C_RESET_PIN_3, OUTPUT);
  digitalWrite(TOF_I2C_RESET_PIN_1, HIGH); // hold all I2Cs in reset mode
  digitalWrite(TOF_I2C_RESET_PIN_2, HIGH); 
  digitalWrite(TOF_I2C_RESET_PIN_3, HIGH); 
  delay(100);

  // I2C communication
  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(1000000); //Sensor has max I2C freq of 400kHz 

  // ToF: initialize sensors and change I2C addresses
  tof_initSensor(&tof_imager_1, TOF_I2C_ADDRESS1, TOF_I2C_RESET_PIN_1);
  tof_initSensor(&tof_imager_2, TOF_I2C_ADDRESS2, TOF_I2C_RESET_PIN_2);
  tof_initSensor(&tof_imager_3, TOF_I2C_ADDRESS3, TOF_I2C_RESET_PIN_3);
  
  // ToF: set sensor resolution and frequency
  tof_imager_1.setResolution(TOF_IMG_RES); //Enable all 64 pads
  tof_imager_2.setResolution(TOF_IMG_RES); //Enable all 64 pads
  tof_imager_3.setResolution(TOF_IMG_RES); //Enable all 64 pads
  tof_imager_1.setRangingFrequency(TOF_IMG_FREQ);
  tof_imager_2.setRangingFrequency(TOF_IMG_FREQ);
  tof_imager_3.setRangingFrequency(TOF_IMG_FREQ);
  
  // ToF: allocate memory for message
  tof_msg_1.meas = (uint16_t*)malloc(sizeof(uint16_t) * TOF_IMG_RES);
  tof_msg_1.meas_length = TOF_IMG_RES;
  tof_msg_2.meas = (uint16_t*)malloc(sizeof(uint16_t) * TOF_IMG_RES);
  tof_msg_2.meas_length = TOF_IMG_RES;
  tof_msg_3.meas = (uint16_t*)malloc(sizeof(uint16_t) * TOF_IMG_RES);
  tof_msg_3.meas_length = TOF_IMG_RES;

  // attach the interrupts
  attachInterrupt(digitalPinToInterrupt(USS_INTERRUPT_PIN_1), uss_interrupt1, CHANGE);
  attachInterrupt(digitalPinToInterrupt(USS_INTERRUPT_PIN_2), uss_interrupt2, CHANGE);
  attachInterrupt(digitalPinToInterrupt(USS_INTERRUPT_PIN_3), uss_interrupt3, CHANGE);
  attachInterrupt(digitalPinToInterrupt(TOF_INTERRUPT_PIN_1), tof_interrupt1, FALLING);
  attachInterrupt(digitalPinToInterrupt(TOF_INTERRUPT_PIN_2), tof_interrupt2, FALLING);
  attachInterrupt(digitalPinToInterrupt(TOF_INTERRUPT_PIN_3), tof_interrupt3, FALLING);

  tof_imager_1.startRanging();
  tof_imager_2.startRanging();
  tof_imager_3.startRanging();

  delay(100);
  Serial.println("Start measurements ...");
}



/*
* ----- MAIN LOOP -----
*/

void loop()
{

  if (uss_interrupt_flag_1) {
    uss_interrupt_flag_1 = false;
    uss_new_meas_flag_1 = true;

    uint32_t echo_duration = uss_echo_end_1 - uss_echo_start_1;
    uss_publishMeas(&uss_pub_1, &uss_msg_1, echo_duration, USS_ID_1, &uss_interrupt_time_1);
#ifdef PRINT_DATA
    uss_printMeas(echo_duration, USS_ID_1);
#endif
  }

  if (uss_interrupt_flag_2) {
    uss_interrupt_flag_2 = false;
    uss_new_meas_flag_2 = true;

    uint32_t echo_duration = uss_echo_end_2 - uss_echo_start_2;
    uss_publishMeas(&uss_pub_2, &uss_msg_2, echo_duration, USS_ID_2, &uss_interrupt_time_2);
#ifdef PRINT_DATA
    uss_printMeas(echo_duration, USS_ID_2);
#endif
  }

  if (uss_interrupt_flag_3) {
    uss_interrupt_flag_3 = false;
    uss_new_meas_flag_3 = true;

    uint32_t echo_duration = uss_echo_end_3 - uss_echo_start_3;
    uss_publishMeas(&uss_pub_3, &uss_msg_3, echo_duration, USS_ID_3, &uss_interrupt_time_3);
#ifdef PRINT_DATA
    uss_printMeas(echo_duration, USS_ID_3);
#endif
  }

  if(tof_interrupt_flag_1) {
    tof_interrupt_flag_1 = false;

    if (tof_imager_1.getRangingData(&tof_data_1)) { //Read distance data into array
      tof_publishMeas(&tof_pub_1, &tof_msg_1, &tof_data_1, TOF_ID_1, &tof_interrupt_time_1);
      tof_printMeas(&tof_data_1, TOF_ID_1);
#ifdef PRINT_DATA
      tof_printMeas(&tof_data_1, TOF_ID_1);
#endif
    }
  }

  if(tof_interrupt_flag_2) {
    tof_interrupt_flag_2 = false;

    if (tof_imager_2.getRangingData(&tof_data_2)) { //Read distance data into array
      tof_publishMeas(&tof_pub_2, &tof_msg_2, &tof_data_2, TOF_ID_2, &tof_interrupt_time_2);
#ifdef PRINT_DATA
      tof_printMeas(&tof_data_2, TOF_ID_2);
#endif
    }
  }

  if(tof_interrupt_flag_3) {
    tof_interrupt_flag_3 = false;

    if (tof_imager_3.getRangingData(&tof_data_3)) { //Read distance data into array
      tof_publishMeas(&tof_pub_3, &tof_msg_3, &tof_data_3, TOF_ID_3, &tof_interrupt_time_3);
#ifdef PRINT_DATA
      tof_printMeas(&tof_data_3, TOF_ID_3);
#endif
    }
  }

  uint32_t trig_duration = micros() - uss_trig_time;
  if ((trig_duration > 5000) && (digitalRead(USS_TRIGGER_PIN) == LOW)) {
    digitalWrite(USS_TRIGGER_PIN, HIGH); 
  }
  if ((trig_duration > USS_TRIG_PERIODE_US)) {
      // && uss_new_meas_flag_1 && uss_new_meas_flag_2 && uss_new_meas_flag_3) {
    // uss_new_meas_flag_1 = false;
    // uss_new_meas_flag_2 = false;
    // uss_new_meas_flag_3 = false;
    uss_trig_time = micros();

    // trigger measurement
    digitalWrite(USS_TRIGGER_PIN, LOW);
    // delay(5); // short delay that sensor detects trigger signal
    // digitalWrite(USS_TRIGGER_PIN, HIGH); 
  } 

  nh.spinOnce();
  //delayMicroseconds(5);
}
