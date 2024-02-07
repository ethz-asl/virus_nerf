#ifndef _ROS_ridgeback_msgs_Status_h
#define _ROS_ridgeback_msgs_Status_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"
#include "ros/duration.h"

namespace ridgeback_msgs
{

  class Status : public ros::Msg
  {
    public:
      typedef std_msgs::Header _header_type;
      _header_type header;
      typedef const char* _hardware_id_type;
      _hardware_id_type hardware_id;
      typedef ros::Duration _mcu_uptime_type;
      _mcu_uptime_type mcu_uptime;
      typedef ros::Duration _connection_uptime_type;
      _connection_uptime_type connection_uptime;
      typedef float _pcb_temperature_type;
      _pcb_temperature_type pcb_temperature;
      typedef float _mcu_temperature_type;
      _mcu_temperature_type mcu_temperature;
      typedef bool _stop_power_status_type;
      _stop_power_status_type stop_power_status;
      typedef bool _stop_engaged_type;
      _stop_engaged_type stop_engaged;
      typedef bool _drivers_active_type;
      _drivers_active_type drivers_active;
      typedef bool _external_stop_present_type;
      _external_stop_present_type external_stop_present;
      typedef bool _charger_connected_type;
      _charger_connected_type charger_connected;
      typedef bool _charging_complete_type;
      _charging_complete_type charging_complete;
      typedef float _measured_battery_type;
      _measured_battery_type measured_battery;
      typedef float _measured_12v_type;
      _measured_12v_type measured_12v;
      typedef float _measured_5v_type;
      _measured_5v_type measured_5v;
      typedef float _measured_inverter_type;
      _measured_inverter_type measured_inverter;
      typedef float _measured_front_axle_type;
      _measured_front_axle_type measured_front_axle;
      typedef float _measured_rear_axle_type;
      _measured_rear_axle_type measured_rear_axle;
      typedef float _measured_light_type;
      _measured_light_type measured_light;
      typedef float _total_current_type;
      _total_current_type total_current;
      typedef float _total_current_peak_type;
      _total_current_peak_type total_current_peak;
      typedef float _total_power_consumed_type;
      _total_power_consumed_type total_power_consumed;

    Status():
      header(),
      hardware_id(""),
      mcu_uptime(),
      connection_uptime(),
      pcb_temperature(0),
      mcu_temperature(0),
      stop_power_status(0),
      stop_engaged(0),
      drivers_active(0),
      external_stop_present(0),
      charger_connected(0),
      charging_complete(0),
      measured_battery(0),
      measured_12v(0),
      measured_5v(0),
      measured_inverter(0),
      measured_front_axle(0),
      measured_rear_axle(0),
      measured_light(0),
      total_current(0),
      total_current_peak(0),
      total_power_consumed(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      uint32_t length_hardware_id = strlen(this->hardware_id);
      varToArr(outbuffer + offset, length_hardware_id);
      offset += 4;
      memcpy(outbuffer + offset, this->hardware_id, length_hardware_id);
      offset += length_hardware_id;
      *(outbuffer + offset + 0) = (this->mcu_uptime.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->mcu_uptime.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->mcu_uptime.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->mcu_uptime.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->mcu_uptime.sec);
      *(outbuffer + offset + 0) = (this->mcu_uptime.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->mcu_uptime.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->mcu_uptime.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->mcu_uptime.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->mcu_uptime.nsec);
      *(outbuffer + offset + 0) = (this->connection_uptime.sec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->connection_uptime.sec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->connection_uptime.sec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->connection_uptime.sec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->connection_uptime.sec);
      *(outbuffer + offset + 0) = (this->connection_uptime.nsec >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->connection_uptime.nsec >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->connection_uptime.nsec >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->connection_uptime.nsec >> (8 * 3)) & 0xFF;
      offset += sizeof(this->connection_uptime.nsec);
      union {
        float real;
        uint32_t base;
      } u_pcb_temperature;
      u_pcb_temperature.real = this->pcb_temperature;
      *(outbuffer + offset + 0) = (u_pcb_temperature.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_pcb_temperature.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_pcb_temperature.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_pcb_temperature.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->pcb_temperature);
      union {
        float real;
        uint32_t base;
      } u_mcu_temperature;
      u_mcu_temperature.real = this->mcu_temperature;
      *(outbuffer + offset + 0) = (u_mcu_temperature.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_mcu_temperature.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_mcu_temperature.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_mcu_temperature.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->mcu_temperature);
      union {
        bool real;
        uint8_t base;
      } u_stop_power_status;
      u_stop_power_status.real = this->stop_power_status;
      *(outbuffer + offset + 0) = (u_stop_power_status.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->stop_power_status);
      union {
        bool real;
        uint8_t base;
      } u_stop_engaged;
      u_stop_engaged.real = this->stop_engaged;
      *(outbuffer + offset + 0) = (u_stop_engaged.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->stop_engaged);
      union {
        bool real;
        uint8_t base;
      } u_drivers_active;
      u_drivers_active.real = this->drivers_active;
      *(outbuffer + offset + 0) = (u_drivers_active.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->drivers_active);
      union {
        bool real;
        uint8_t base;
      } u_external_stop_present;
      u_external_stop_present.real = this->external_stop_present;
      *(outbuffer + offset + 0) = (u_external_stop_present.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->external_stop_present);
      union {
        bool real;
        uint8_t base;
      } u_charger_connected;
      u_charger_connected.real = this->charger_connected;
      *(outbuffer + offset + 0) = (u_charger_connected.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->charger_connected);
      union {
        bool real;
        uint8_t base;
      } u_charging_complete;
      u_charging_complete.real = this->charging_complete;
      *(outbuffer + offset + 0) = (u_charging_complete.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->charging_complete);
      union {
        float real;
        uint32_t base;
      } u_measured_battery;
      u_measured_battery.real = this->measured_battery;
      *(outbuffer + offset + 0) = (u_measured_battery.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_battery.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_battery.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_battery.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_battery);
      union {
        float real;
        uint32_t base;
      } u_measured_12v;
      u_measured_12v.real = this->measured_12v;
      *(outbuffer + offset + 0) = (u_measured_12v.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_12v.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_12v.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_12v.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_12v);
      union {
        float real;
        uint32_t base;
      } u_measured_5v;
      u_measured_5v.real = this->measured_5v;
      *(outbuffer + offset + 0) = (u_measured_5v.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_5v.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_5v.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_5v.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_5v);
      union {
        float real;
        uint32_t base;
      } u_measured_inverter;
      u_measured_inverter.real = this->measured_inverter;
      *(outbuffer + offset + 0) = (u_measured_inverter.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_inverter.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_inverter.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_inverter.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_inverter);
      union {
        float real;
        uint32_t base;
      } u_measured_front_axle;
      u_measured_front_axle.real = this->measured_front_axle;
      *(outbuffer + offset + 0) = (u_measured_front_axle.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_front_axle.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_front_axle.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_front_axle.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_front_axle);
      union {
        float real;
        uint32_t base;
      } u_measured_rear_axle;
      u_measured_rear_axle.real = this->measured_rear_axle;
      *(outbuffer + offset + 0) = (u_measured_rear_axle.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_rear_axle.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_rear_axle.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_rear_axle.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_rear_axle);
      union {
        float real;
        uint32_t base;
      } u_measured_light;
      u_measured_light.real = this->measured_light;
      *(outbuffer + offset + 0) = (u_measured_light.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_measured_light.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_measured_light.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_measured_light.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->measured_light);
      union {
        float real;
        uint32_t base;
      } u_total_current;
      u_total_current.real = this->total_current;
      *(outbuffer + offset + 0) = (u_total_current.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_total_current.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_total_current.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_total_current.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->total_current);
      union {
        float real;
        uint32_t base;
      } u_total_current_peak;
      u_total_current_peak.real = this->total_current_peak;
      *(outbuffer + offset + 0) = (u_total_current_peak.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_total_current_peak.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_total_current_peak.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_total_current_peak.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->total_current_peak);
      offset += serializeAvrFloat64(outbuffer + offset, this->total_power_consumed);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      uint32_t length_hardware_id;
      arrToVar(length_hardware_id, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_hardware_id; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_hardware_id-1]=0;
      this->hardware_id = (char *)(inbuffer + offset-1);
      offset += length_hardware_id;
      this->mcu_uptime.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->mcu_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->mcu_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->mcu_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->mcu_uptime.sec);
      this->mcu_uptime.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->mcu_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->mcu_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->mcu_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->mcu_uptime.nsec);
      this->connection_uptime.sec =  ((uint32_t) (*(inbuffer + offset)));
      this->connection_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->connection_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->connection_uptime.sec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->connection_uptime.sec);
      this->connection_uptime.nsec =  ((uint32_t) (*(inbuffer + offset)));
      this->connection_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->connection_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->connection_uptime.nsec |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->connection_uptime.nsec);
      union {
        float real;
        uint32_t base;
      } u_pcb_temperature;
      u_pcb_temperature.base = 0;
      u_pcb_temperature.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_pcb_temperature.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_pcb_temperature.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_pcb_temperature.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->pcb_temperature = u_pcb_temperature.real;
      offset += sizeof(this->pcb_temperature);
      union {
        float real;
        uint32_t base;
      } u_mcu_temperature;
      u_mcu_temperature.base = 0;
      u_mcu_temperature.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_mcu_temperature.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_mcu_temperature.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_mcu_temperature.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->mcu_temperature = u_mcu_temperature.real;
      offset += sizeof(this->mcu_temperature);
      union {
        bool real;
        uint8_t base;
      } u_stop_power_status;
      u_stop_power_status.base = 0;
      u_stop_power_status.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->stop_power_status = u_stop_power_status.real;
      offset += sizeof(this->stop_power_status);
      union {
        bool real;
        uint8_t base;
      } u_stop_engaged;
      u_stop_engaged.base = 0;
      u_stop_engaged.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->stop_engaged = u_stop_engaged.real;
      offset += sizeof(this->stop_engaged);
      union {
        bool real;
        uint8_t base;
      } u_drivers_active;
      u_drivers_active.base = 0;
      u_drivers_active.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->drivers_active = u_drivers_active.real;
      offset += sizeof(this->drivers_active);
      union {
        bool real;
        uint8_t base;
      } u_external_stop_present;
      u_external_stop_present.base = 0;
      u_external_stop_present.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->external_stop_present = u_external_stop_present.real;
      offset += sizeof(this->external_stop_present);
      union {
        bool real;
        uint8_t base;
      } u_charger_connected;
      u_charger_connected.base = 0;
      u_charger_connected.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->charger_connected = u_charger_connected.real;
      offset += sizeof(this->charger_connected);
      union {
        bool real;
        uint8_t base;
      } u_charging_complete;
      u_charging_complete.base = 0;
      u_charging_complete.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->charging_complete = u_charging_complete.real;
      offset += sizeof(this->charging_complete);
      union {
        float real;
        uint32_t base;
      } u_measured_battery;
      u_measured_battery.base = 0;
      u_measured_battery.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_battery.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_battery.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_battery.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_battery = u_measured_battery.real;
      offset += sizeof(this->measured_battery);
      union {
        float real;
        uint32_t base;
      } u_measured_12v;
      u_measured_12v.base = 0;
      u_measured_12v.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_12v.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_12v.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_12v.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_12v = u_measured_12v.real;
      offset += sizeof(this->measured_12v);
      union {
        float real;
        uint32_t base;
      } u_measured_5v;
      u_measured_5v.base = 0;
      u_measured_5v.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_5v.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_5v.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_5v.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_5v = u_measured_5v.real;
      offset += sizeof(this->measured_5v);
      union {
        float real;
        uint32_t base;
      } u_measured_inverter;
      u_measured_inverter.base = 0;
      u_measured_inverter.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_inverter.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_inverter.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_inverter.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_inverter = u_measured_inverter.real;
      offset += sizeof(this->measured_inverter);
      union {
        float real;
        uint32_t base;
      } u_measured_front_axle;
      u_measured_front_axle.base = 0;
      u_measured_front_axle.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_front_axle.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_front_axle.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_front_axle.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_front_axle = u_measured_front_axle.real;
      offset += sizeof(this->measured_front_axle);
      union {
        float real;
        uint32_t base;
      } u_measured_rear_axle;
      u_measured_rear_axle.base = 0;
      u_measured_rear_axle.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_rear_axle.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_rear_axle.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_rear_axle.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_rear_axle = u_measured_rear_axle.real;
      offset += sizeof(this->measured_rear_axle);
      union {
        float real;
        uint32_t base;
      } u_measured_light;
      u_measured_light.base = 0;
      u_measured_light.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_measured_light.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_measured_light.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_measured_light.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->measured_light = u_measured_light.real;
      offset += sizeof(this->measured_light);
      union {
        float real;
        uint32_t base;
      } u_total_current;
      u_total_current.base = 0;
      u_total_current.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_total_current.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_total_current.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_total_current.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->total_current = u_total_current.real;
      offset += sizeof(this->total_current);
      union {
        float real;
        uint32_t base;
      } u_total_current_peak;
      u_total_current_peak.base = 0;
      u_total_current_peak.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_total_current_peak.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_total_current_peak.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_total_current_peak.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->total_current_peak = u_total_current_peak.real;
      offset += sizeof(this->total_current_peak);
      offset += deserializeAvrFloat64(inbuffer + offset, &(this->total_power_consumed));
     return offset;
    }

    virtual const char * getType() override { return "ridgeback_msgs/Status"; };
    virtual const char * getMD5() override { return "5b3d8e0f8c2c371cf7df823649f67044"; };

  };

}
#endif
