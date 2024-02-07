import rospy

from pcl_tools.pcl_saver import PCLSaver



def main():
    pcl_saver = PCLSaver(
        topic_pcl=rospy.get_param('topic_pcl'),
        topic_pose=rospy.get_param('topic_pose'),
        save_dir=rospy.get_param('save_dir'),
    )
    pcl_saver.subscribe()

if __name__ == "__main__":
    main()