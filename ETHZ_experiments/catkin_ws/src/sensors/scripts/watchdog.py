#!/usr/bin/env python
import rospy
import rostopic  

class WatchInfo():
    def __init__(
        self,
        topic:str,
        freq:float=4.8,
        error_max:float=0.2,
        std_max:float=0.1,
    ) -> None:
        """
        Args:
            topic: topic name: str
            freq: expected frequency; float
            error_max: maximum allowed error; float
            std_max; maximum allowed standard deviation; float
        """
        self.topic = topic
        self.freq = freq
        self.error_max = error_max
        self.std_max = std_max


class Watchdog():
    def __init__(
        self,
        watch_list:list,
        watchdog_rate:float=0.2,
    ) -> None:
        """_summary_

        Args:
            watch_list: list of topics to watch; list of WatchInfo
            watchdog_rate: watchdog rate in hz; float
        """
        self.freqs = [watch_info.freq for watch_info in watch_list]
        self.error_max = [watch_info.error_max for watch_info in watch_list]
        self.std_max = [watch_info.std_max for watch_info in watch_list]
        self.watchdog_rate = watchdog_rate
        
        rospy.loginfo("\n\n----------\nINIT WATCHDOG\n----------\n\n")
        
        # ROS
        self.topics = [watch_info.topic for watch_info in watch_list]
        self.subscribes = []
        rospy.init_node('watchdog_node', anonymous=True)
    
    def subscribe(
        self,
    ):
        h = rostopic.ROSTopicHz(-1)  
        for topic in self.topics:
            self.subscribes.append(rospy.Subscriber(topic, rospy.AnyMsg, h.callback_hz, callback_args=topic))
        
        # initialize watchdog rate  
        rate = rospy.Rate(self.watchdog_rate) # ROS Rate at 1Hz
        while not rospy.is_shutdown():
            
            # watch all topics
            for i, topic in enumerate(self.topics):
                msg = h.get_hz(topic)
                
                if msg == None:
                    rospy.logwarn(f"Watchdog.subscribe: topic: {topic} not published")
                else:
                    freq = msg[0]
                    std = msg [3]
                    
                    # verify frequency
                    if abs(freq-self.freqs[i]) > self.error_max[i]:
                        rospy.logwarn(f"Watchdog.subscribe: topic: {topic}, \n        "
                                      f"freq: {freq:.6}, "
                                      f"abs. error: {abs(freq-self.freqs[i]):.6} is too large!")
                        
                    # verify std
                    if std > self.std_max[i]:
                        rospy.logwarn(f"Watchdog.subscribe: topic: {topic}, \n        "
                                      f"std: {std:.6}, is larger than max: {self.std_max[i]:.6}")
            rate.sleep()


def main():
    dog = Watchdog(
        watch_list=[
            WatchInfo(
                topic="/TOF1",
                freq=13.0,
                error_max=0.5,
            ),
            WatchInfo(
                topic="/TOF3",
                freq=13.0,
                error_max=0.5,
            ),
            WatchInfo(
                topic="/USS1",
                freq=5.0,
            ),
            WatchInfo(
                topic="/USS3",
                freq=5.0,
            ),
            WatchInfo(
                topic="/CAM1/aligned_depth_to_color/image_raw",
            ),
            WatchInfo(
                topic="/CAM3/aligned_depth_to_color/image_raw",
            ),
            WatchInfo(
                topic="/CAM1/color/image_raw",
            ),
            WatchInfo(
                topic="/CAM3/color/image_raw",
            ),
            WatchInfo(
                topic="/CAM1/color/camera_info",
            ),
            WatchInfo(
                topic="/CAM3/color/camera_info",
            ),
            WatchInfo(
                topic="/CAM1/color/metadata",
            ),
            WatchInfo(
                topic="/CAM3/color/metadata",
            ),
        ],
    )
    dog.subscribe()

if __name__ == '__main__':
    main()