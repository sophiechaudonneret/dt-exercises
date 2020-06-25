# Package `explicit_coordinator` {#explicit_coordinator}

<move-here src='#explicit_coordinator-autogenerated'/>


## In-depth

In order to launch the coordination node, run the following command:

    $ roslaunch explicit_coordination coordination_node.launch veh:=![robot name]

This launches the coordination node. In order to be able to use the coordination node it is also needed to launch the detection node and led emitter node (look in the corresponding packages for instructions).
The coordination is based on different states in which the Duckiebot might find itself in. This can be read by running:

    $ rotopic echo ![robot name]/coordination_node/coordination_state

In order to launch the full explicit coordination system (i.e., with detection and emission) please run:

    $ roslaunch explicit_coordination general.launch veh:=![robot name]

Note that the apriltags_node needs to be started in order to observe successfull behavior of the Duckiebot.
In order to launch the full system you may run:

    $ cd ~/duckietown
    $ make coordination2017

Note that this will make the whole system work.