#ifndef _ROS_sensors_TOF_h
#define _ROS_sensors_TOF_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace sensors
{

  class TOF : public ros::Msg
  {
    public:
      typedef std_msgs::Header _header_type;
      _header_type header;
      uint32_t meas_length;
      typedef uint16_t _meas_type;
      _meas_type st_meas;
      _meas_type * meas;

    TOF():
      header(),
      meas_length(0), st_meas(), meas(nullptr)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      *(outbuffer + offset + 0) = (this->meas_length >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->meas_length >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->meas_length >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->meas_length >> (8 * 3)) & 0xFF;
      offset += sizeof(this->meas_length);
      for( uint32_t i = 0; i < meas_length; i++){
      *(outbuffer + offset + 0) = (this->meas[i] >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->meas[i] >> (8 * 1)) & 0xFF;
      offset += sizeof(this->meas[i]);
      }
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      uint32_t meas_lengthT = ((uint32_t) (*(inbuffer + offset))); 
      meas_lengthT |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1); 
      meas_lengthT |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2); 
      meas_lengthT |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3); 
      offset += sizeof(this->meas_length);
      if(meas_lengthT > meas_length)
        this->meas = (uint16_t*)realloc(this->meas, meas_lengthT * sizeof(uint16_t));
      meas_length = meas_lengthT;
      for( uint32_t i = 0; i < meas_length; i++){
      this->st_meas =  ((uint16_t) (*(inbuffer + offset)));
      this->st_meas |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->st_meas);
        memcpy( &(this->meas[i]), &(this->st_meas), sizeof(uint16_t));
      }
     return offset;
    }

    virtual const char * getType() override { return "sensors/TOF"; };
    virtual const char * getMD5() override { return "fd28ad540c9e3978921180a82c8d9d8b"; };

  };

}
#endif
