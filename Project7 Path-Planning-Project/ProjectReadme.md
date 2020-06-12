# Path Planning Project
Udacity Self-Driving Car Nanodegree - Path Planning Project
## Generating Paths
The path planning algorithm starts in the file [src/main.cpp](./src/main.cpp#L98)
### Perception and Fusion [line 98 to line 156](./src/main.cpp#L98)
In this part of the code using sensor fusion we determine on which lane the other from sensor fusion detected cars are driving. 
This is done by calculating the distance d from the middle lane which separates both traffic directions and considering a lane width of 4. 
After identifying on which lane the other cars are driving, we consider a distance of 30 meters to judge if the identified vehicles
are too close to the ego vehicle.
### Behavior [line 168 to line 202](./src/main.cpp#L168)
The behavior part decides what to do based on the perception and fusion information. For example, if to decrease speed when a car is in 
front of the ego vehicle or to change lane if it is possible.
### Trajectory [line 206 to line 317](./src/main.cpp#L206)
In this part we calculate the trajectory of the ego vehicle based on the information from behavior, the ego vehicle coordinates and the past
path points.
To calculate the trajectory we introduce a spline function with three points with a distance of 30m for each point to initialize the spline 
and using the previous trajectory or the car position in conjunction to the three points. To ensure continuity we copy the last trajectory
points to the new trajectory. On every trajectory points we decide based on the behavior information to increase or decrease the speed of the
ego vehicle.

