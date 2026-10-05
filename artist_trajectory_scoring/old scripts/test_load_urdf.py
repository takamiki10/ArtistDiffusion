from yourdfpy import URDF

URDF_PATH = "robot_model/rokae_ros_ws/rokae_ros_pkg/src/rokae_xMateCR7_moveit_config/config/gazebo_xMateCR7.urdf"

robot = URDF.load(URDF_PATH, load_meshes=False)
print("Robot name:", robot.robot.name)
print("Number of joints:", len(robot.robot.joints))
print("\nJoints:")
for j in robot.robot.joints:
    print(f"  {j.name:30s} type={j.type:10s} parent={j.parent:25s} child={j.child}")
