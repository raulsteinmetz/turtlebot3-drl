# Deep Reinforcement Learning applied to the TurtleBot3 Navigation Task

This project offers a framework for training Deep Reinforcement Learning algorithms in the TurtleBot3 navigation task.

## Features
- Multiple environment setups
- Multiple DRL algorithms (model-free and model-based)

## Configuring ROS packages

To configure the Deep Reinforcement Learning (DRL) stages on gazebo, follow these steps:

1. **Install a ROS2 distribution (foxy and humble are recommended)**
 
   [Follow this tutorial for foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
   
   [Follow this tutorial for humble](https://docs.ros.org/en/humble/Installation.html)
   

2. **Configure robotis gazebo package**

   [Follow this tutorial](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

2. **Create a backup folder for your old `launch`, `models`, and `worlds`, if necessary**

   ```bash
   mkdir ~/backup
   ```

4. **Copy the contents of the `launch`, `models`, and `worlds` directories into your robotis turtlebot3 environment:**

   Only change the `your_ros_distro` in the beginning of each `mv` command. It should be the folder you created according to step 1 of this tutorial (besides your ros distro name, it can be turtlebot_ws or even some custom name you created when following the tutorial).

   - For `launch`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch/ ~/backup/
     cp -r turtlebot3_gazebo/launch ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch
     ```
   
   - For `models`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models/ ~/backup/
     cp -r turtlebot3_gazebo/models ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models
     ```
   
   - For `worlds`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/ ~/backup
     cp -r turtlebot3_gazebo/worlds ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
     ```

5. **Build your project to apply the changes**
   ```bash
   cd ~/your_ros_distro/
   colcon build --symlink-install
   ```

## Training Your Agent
- Run the gazebo simulation:

  `ros2 launch turtlebot3_gazebo turtle_stage<number>.py`

- To train the model-free agent:
  
   `python3 train.py --agent <name> --stage <number>`
  
- To train Dreamer v3:
  
   `python3 dreamer.py --configs turtle --task turtle --logdir ./logdir/turtle (TODO: add stage parameter)`

## Plot learning curve

`python3 learning_curve.py --agent <name> --stage <number>`

## Testing Your Agent
- Run the gazebo simulation:

   `ros2 launch turtlebot3_gazebo turtle_stage<number>.py`

- To test the model-free agent:
  
   `python3 test.py --agent <name> --stage <number>`

 ## Moving the model you've just trained to the best models folder
  
   `python3 save_to_best.py --agent <name> --stage <number>`


## Folder Structure
- best_models: Contains our best agent models and their training and testing logs
- models: Folder where current training models and logs are saved
- turtle_env/turtle.py: Contains the gym environment (uses ROS2 in the background)
- turtlebot3_gazebo: Contains files to set up the gazebo environment
- model_free: Contains the model-free agents' code
- dreamerv3 torch: Contains the Dreamer v3 algorithm code
- plots: Contains the learning curve plots for the model-free agents

