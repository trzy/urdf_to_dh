# URDF to Denavit-Hartenberg Parameters

This is [Andy McEvoy's aweseome utility](https://github.com/mcevoyandy/urdf_to_dh) but with all the ROS bullshit stripped out and an additional option to generate a simplified URDF file consisting only of non-fixed joints. I needed the latter to simplify a URDF description of a robot arm consisting only of revolute joints.
Please see the [original repository](https://github.com/mcevoyandy/urdf_to_dh) for documentation.

To run the utility:

```
python -m urdf_to_dh.generate_dh crane_x7.urdf
```

To generate a simplified URDF file:

```
python -m urdf_to_dh.generate_dh crane_x7.urdf --simplified-urdf=crane_x7_simple.urdf
```
