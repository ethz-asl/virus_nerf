import time
import os
from serial import Serial
import numpy as np
import pandas as pd


def readserial(comport, baudrate, num_meas):

    ser = Serial(comport, baudrate, timeout=0.1) # 1/timeout is the frequency at which the port is read

    time.sleep(1) # wait for the serial connection to initialize

    # ignore first 20 lines
    for _ in range(30):
        data = ser.readline().decode().strip()
        print(f"Ignore: {data}")

    meas = []
    while len(meas) < num_meas:
        data = ser.readline().decode().strip()
        print(f"{data}")
        if data[5:] == '':
            continue
        meas.append(float(data[5:]))

    ser.close()
    return np.array(meas)

def convertMeas(meas, sensor):
    if sensor == "MB1603":
        return meas / 1000
    elif sensor == "URM37":
        return meas / 100
    elif sensor == "HC-SR04":
        return meas / 1000


def main():
    # measurement parameters
    sensor = "HC-SR04" #"MB1603" #"URM37" #"HC-SR04" # 
    dist = 2
    angle = 0
    num_meas = 200

    # load data frame
    file_path = os.path.join("data", sensor+".csv")
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()
        
    # make measurement
    meas = readserial('COM8', 9600, num_meas=num_meas)      
    meas = convertMeas(meas, sensor)

    # add measurmenet to data frame
    df[f"{dist}m_{angle}deg"] = meas

    # save data frame
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()
    