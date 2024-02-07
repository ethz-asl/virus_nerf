#ifndef _ROS_sensors_uss_h
#define _ROS_sensors_uss_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"
#include "std_msgs/Header.h"

namespace sensors
{

  class uss : public ros::Msg
  {
    public:
      typedef std_msgs::Header _header_type;
      _header_type header;
      typedef uint32_t _meas_type;
      _meas_type meas;

    uss():
      header(),
      meas(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      offset += this->header.serialize(outbuffer + offset);
      *(outbuffer + offset + 0) = (this->meas >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->meas >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (this->meas >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (this->meas >> (8 * 3)) & 0xFF;
      offset += sizeof(this->meas);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      offset += this->header.deserialize(inbuffer + offset);
      this->meas =  ((uint32_t) (*(inbuffer + offset)));
      this->meas |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      this->meas |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      this->meas |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      offset += sizeof(this->meas);
     return offset;
    }

    virtual const char * getType() override { return "sensors/uss"; };
    virtual const char * getMD5() override { return "a54e8d83083240737bbd4b60f3cbdd28"; };

  };

}
#endif
