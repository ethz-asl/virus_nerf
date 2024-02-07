#ifndef _ROS_ridgeback_msgs_RGB_h
#define _ROS_ridgeback_msgs_RGB_h

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace ridgeback_msgs
{

  class RGB : public ros::Msg
  {
    public:
      typedef float _red_type;
      _red_type red;
      typedef float _green_type;
      _green_type green;
      typedef float _blue_type;
      _blue_type blue;

    RGB():
      red(0),
      green(0),
      blue(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_red;
      u_red.real = this->red;
      *(outbuffer + offset + 0) = (u_red.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_red.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_red.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_red.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->red);
      union {
        float real;
        uint32_t base;
      } u_green;
      u_green.real = this->green;
      *(outbuffer + offset + 0) = (u_green.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_green.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_green.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_green.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->green);
      union {
        float real;
        uint32_t base;
      } u_blue;
      u_blue.real = this->blue;
      *(outbuffer + offset + 0) = (u_blue.base >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (u_blue.base >> (8 * 1)) & 0xFF;
      *(outbuffer + offset + 2) = (u_blue.base >> (8 * 2)) & 0xFF;
      *(outbuffer + offset + 3) = (u_blue.base >> (8 * 3)) & 0xFF;
      offset += sizeof(this->blue);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      union {
        float real;
        uint32_t base;
      } u_red;
      u_red.base = 0;
      u_red.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_red.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_red.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_red.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->red = u_red.real;
      offset += sizeof(this->red);
      union {
        float real;
        uint32_t base;
      } u_green;
      u_green.base = 0;
      u_green.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_green.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_green.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_green.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->green = u_green.real;
      offset += sizeof(this->green);
      union {
        float real;
        uint32_t base;
      } u_blue;
      u_blue.base = 0;
      u_blue.base |= ((uint32_t) (*(inbuffer + offset + 0))) << (8 * 0);
      u_blue.base |= ((uint32_t) (*(inbuffer + offset + 1))) << (8 * 1);
      u_blue.base |= ((uint32_t) (*(inbuffer + offset + 2))) << (8 * 2);
      u_blue.base |= ((uint32_t) (*(inbuffer + offset + 3))) << (8 * 3);
      this->blue = u_blue.real;
      offset += sizeof(this->blue);
     return offset;
    }

    virtual const char * getType() override { return "ridgeback_msgs/RGB"; };
    virtual const char * getMD5() override { return "fc84fca2ee69069d6d5c4147f9b2e33a"; };

  };

}
#endif
