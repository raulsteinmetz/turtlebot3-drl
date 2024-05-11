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

3. **Copy the contents of the `launch`, `models`, and `worlds` directories into your robotis turtlebot3 environment:**

   - For `launch`:
     ```bash
     rm -rf ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch/*
     cp -r turtlebot3_gazebo/launch ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch
     ```
   
   - For `models`:
     ```bash
     rm -rf ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models/*
     cp -r turtlebot3_gazebo/models ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models
     ```
   
   - For `worlds`:
     ```bash
     rm -rf ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/*
     cp -r turtlebot3_gazebo/worlds ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
     ```

4. **Build your project to apply the changes**
   ```bash
   cd ~/your_ros_distro/
   colcon build --symlink-install
   ```

## Training Your Agent
- Run the gazebo simulation:

  `ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage<stage>.launch.py`, set gazebo physics real time update to 5000 (5x faster) to reproduce our results!

- To train the model-free agent:
  
   `python3 train.py --agent (ddpg or sac or sac_x or sac_x_hybrid or td3) --stage (from 1 to 4 currently)`
- To train Dreamer v3:
  
   `python3 dreamer.py --configs turtle --task turtle --logdir ./logdir/turtle (TODO: add stage parameter)`

## Plot learning curve

`python3 learning_curve.py --agent (ddpg or sac or sac_x or sac_x_hybrid or td3) --stage (from 1 to 4 currently)`

## Testing Your Agent
- Run the gazebo simulation:

  `ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage<stage>.launch.py`, set gazebo physics real time update to 5000 (5x faster) to reproduce our results!

- To test the model-free agent:
  
   `python3 test.py --agent (ddpg or sac or sac_x or sac_x_hybrid or td3) --stage (from 1 to 4 currently)`

 ## Moving the model you've just trained to the best models folder
  
   `python3 save_to_best.py --agent (ddpg or sac or sac_x or sac_x_hybrid or td3) --stage (from 1 to 4 currently)`


## Folder Structure
- best_models: Contains our best agent models and their training and testing logs
- models: Folder where current training models and logs are saved
- turtle_env/turtle.py: Contains the gym environment (uses ROS2 in the background)
- turtlebot3_gazebo: Contains files to set up the gazebo environment
- model_free: Contains the model-free agents' code
- dreamerv3 torch: Contains the Dreamer v3 algorithm code
- plots: Contains the learning curve plots for the model-free agents

