// # Editor     : roker
// # Date       : 05.03.2018

// # Product name: URM V5.0 ultrasonic sensor
// # Product SKU : SEN0001
// # Version     : 1.0

// # Description:
// # The sketch for using the URM37 Serial  mode from DFRobot
// #   and writes the values to the serialport

// # Connection:
// #       Vcc (Arduino)      -> Pin 1 VCC (URM V5.0)
// #       GND (Arduino)      -> Pin 2 GND (URM V5.0)
// #       Pin TX1 (Arduino)  -> Pin 8 RXD (URM V5.0)
// #       Pin RX0 (Arduino)  -> Pin 9 TXD (URM V5.0)
// # Working Mode: Serial  Mode.

// uint8_t cmd[4] = {0x11, 0x00, 0x00, 0x11}; // temperature measure command
uint8_t cmd[4] = {0x22, 0x00, 0x00, 0x22}; // distance measure command



void setup()
{
  Serial.begin(9600);
  delay(100);
  Serial.println("Init the sensor");
}


void loop()
{
  SerialCmd();
  delay(200);
}

void SerialCmd()
{

  for(int i=0; i<4; i++) {
    Serial.write(cmd[i]);
  }

  while(Serial.available() > 0)  // if received data
  { 
    // read data
    uint8_t data[4];
    for(int i=0; i<4; i++) {
      data[i] = Serial.read();
    }

    // verify check sum
    int16_t val = 0xFFFF;
    if(data[3] == data[0]+data[1]+data[2]) {

      // convert measurement
      switch(data[0]) {
        case 0x11:
          val = convertTemp(data);
          break;
        case 0x22:
          val = convertDist(data);
          break;
        default:
          val = 0xFFFF;
          break;
      }
    }

    switch(data[0]) {
        case 0x11:
          Serial.print("temperature : ");
          Serial.print(val, DEC);
          Serial.println(" oC");
          break;
        case 0x22:
          Serial.print("distance : ");
          Serial.print(val, DEC);
          Serial.println(" cm");
          break;
        default:
          Serial.println("Mode not available");
          break;
      }

    

  }
}

int16_t convertTemp(uint8_t *data)
{
  // verify if measurement is valid
  if(data[1]==0xFF && data[2]==0xFF) {
    return 0xFFFF;
  }

  // temperature is given by 12bits: 
  // 4 lower bits of data[1] and 8 bits of data[2]
  int16_t temp = data[2];
  temp += (data[1] & 0x0F) << 8;

  // temperature is negative if first four bits are 1
  if((data[1] & 0xF0) > 0) {
    temp *= -1;
  }

  return temp;
}

int16_t convertDist(uint8_t *data)
{
  // verify if measurement is valid
  if(data[1]==0xFF && data[2]==0xFF) {
    return 0xFFFF;
  }

  // distance is given by 16bits: 
  // 8 bits of data[1] and 8 bits of data[2]
  int16_t dist = data[2];
  dist += data[1] << 8;

  return dist;
}

