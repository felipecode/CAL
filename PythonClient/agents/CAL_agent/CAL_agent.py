import os, math, time
import scipy
import tensorflow as tf
import numpy as np
from carla.agent import Agent
from carla.carla_server_pb2 import Control
from carla.planner.map import CarlaMap
import logging

# own imports
from plans import Centerlines
<<<<<<< HEAD
from controller import PID
from perception import CAL_network

# set up the neural net
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.visible_device_list = '0'
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.35
sess = tf.Session(config=config_gpu)

# choose model
NEURAL_NET = CAL_network()

### PARAMS
MAX_STEER = math.radians(35.0)

def get_params_from_txt(path):
    return np.loadtxt(path,comments='#', delimiter=',')
=======
from perception import CAL_network
from controller import PID

# maximum steering angle, limited by the car model
MAX_STEER = math.radians(35.0)

def get_params_from_txt(path):
    return np.loadtxt(path, comments='#', delimiter=',')
>>>>>>> upstream/master

def cycle_signal(signal, value):
    # update a given signal with newest value
    signal.pop(0)
    signal.append(value)

class VehicleState(object):
    """
    class containing the vehicle state
<<<<<<< HEAD
    == signals that are important to retain in memory
=======
    includes all signals that are important to retain in memory
>>>>>>> upstream/master
    """
    def __init__(self ):
        self.speed = 0
        self.directions_list = {0}
        self.speed_limit = 30
        self.direction = 0
        self.center_distance_GT = 0
<<<<<<< HEAD
        self.steer_hist = [0]*4 # default *4
        self.direction_hist = [np.nan]*20 # default *20
        self.image_hist = None
        self.last_steer = 0
=======
        self.image_hist = None
        self.standing_to_long = False
>>>>>>> upstream/master

class Timer(object):
    def __init__(self):
        self._lap_time = time.time()
        self._refractory_time = False

    def in_refractory(self):
<<<<<<< HEAD
        return self._refractory_time
=======
	    return self._refractory_time
>>>>>>> upstream/master

    def elapsed_seconds(self):
        return time.time() - self._lap_time

    def reset_lap_time(self):
        self._lap_time = time.time()

class CAL(Agent):

    def __init__(self, city_name):
        self.timer = Timer()

<<<<<<< HEAD
        # Map and centerlines setup
=======
        # Map setup
>>>>>>> upstream/master
        self._map = CarlaMap(city_name)
        self._centerlines = Centerlines(city_name)

        # Agent Setup
<<<<<<< HEAD
        Agent.__init__(self)
        self._neural_net = NEURAL_NET
        self._seq_len = NEURAL_NET.model.input_shape[0][1]
        self._state = VehicleState()
        self._agents_present = False

        # Controller setup
        param_path = os.path.dirname(__file__) + '/controller/params/'
        cruise_params = get_params_from_txt(param_path + 'PID_cruise_params.txt')
        self._PID_cruise = PID(*cruise_params)
        follow_params = get_params_from_txt(param_path + 'PID_follow_params.txt')
=======
        Agent.__init__(self)        
        self._neural_net = CAL_network()
        self._seq_len = self._neural_net.model.max_input_shape
        self._state = VehicleState()
        self._agents_present = False
       
        # Controller setup
        param_path = os.path.dirname(__file__) + '/controller/params/'
        cruise_params = get_params_from_txt(param_path + 'cruise_params.txt')
        self._PID_cruise = PID(*cruise_params)
        follow_params = get_params_from_txt(param_path + 'follow_params.txt')
>>>>>>> upstream/master
        self._PID_follow = PID(*follow_params)

        # General Parameter setup
        general_params = get_params_from_txt(param_path + 'general_params.txt')
<<<<<<< HEAD
        self.c, self.d = general_params[1], general_params[2]
        self.Kl_STANLEY = general_params[3]
        self.Kr_STANLEY = general_params[4]
        self.K0_STANLEY = general_params[5]
        self.curve_slowdown = general_params[6]
        self.DELTAl = general_params[7]
        self.DELTAr = general_params[8]
        self.DELTA0 = general_params[9]
        self.EXP_DECAY = general_params[10]
=======
        self.c, self.d = general_params[0], general_params[1]
        self.Kl_STANLEY = general_params[2]
        self.Kr_STANLEY = general_params[3]
        self.K0_STANLEY = general_params[4]
        self.curve_slowdown = general_params[5]
        self.DELTAl = general_params[6]
        self.DELTAr = general_params[7]
        self.DELTA0 = general_params[8]
        self.EXP_DECAY = general_params[9]
>>>>>>> upstream/master

    def reset_state(self):
        """ for resetting at the start of a new episode"""
        self._state = VehicleState()

    def run_step(self, measurements, sensor_data, carla_direction, target):
        # update the vehicle state
        self._state.speed = measurements.player_measurements.forward_speed*3.6
<<<<<<< HEAD
        # get the current location and orientation of the agent in the world COS
        location, psi = self._get_location_and_orientation(measurements)

        # uncomment the next line if you want to speed up the benchmark
        # we dont need to stop for red lights, if  there are no cars present
=======
        
        # get the current location and orientation of the agent in the world COS
        location, psi = self._get_location_and_orientation(measurements)

	    # check if there are other cars or pedestrians present
	    # speed up the benchmark, because we dont need to stop for red lights
>>>>>>> upstream/master
        self._agents_present = any([agent.HasField('vehicle') for agent in measurements.non_player_agents])

        # get the possible directions (at position of the front axle)
        front_axle_pos = self._get_front_axle(location, psi)
<<<<<<< HEAD
        directions_list_new = self._centerlines.get_directions(front_axle_pos)
=======
        try:
            directions_list_new = self._centerlines.get_directions(front_axle_pos)
        except:
            directions_list_new = {}
>>>>>>> upstream/master

        # determine the current direction
        if directions_list_new:
            self._set_current_direction(directions_list_new, carla_direction)

        #### SIGNALS FOR EVALUATION ####
        # set the correct center line and get the ground truth
        self._state.center_distance_GT = self._centerlines.get_center_distance(front_axle_pos)
        self._state.long_accel, self._state.lat_accel = self._get_accel(measurements, psi)
        ################################

        # cycle the image sequence
        new_im = sensor_data['CameraRGB'].data
        if self._state.image_hist is None:
            im0 = self._neural_net.preprocess_image(new_im, sequence=True)
            self._state.image_hist = im0.repeat(self._seq_len, axis=1)
        else:
            # get the newest entry
            im0 = self._neural_net.preprocess_image(new_im, sequence=True)
<<<<<<< HEAD
            # drop the oldest entry
=======
            # drop the oldelst entry
>>>>>>> upstream/master
            self._state.image_hist = self._state.image_hist[:,1:,:,:]
            # add new entry
            self._state.image_hist = np.concatenate((self._state.image_hist, im0), axis=1)

        # calculate the control
        control = self._compute_action(carla_direction, self._state.direction)
<<<<<<< HEAD
        cycle_signal(self._state.direction_hist, self._state.direction)
=======
>>>>>>> upstream/master

        return control

    def _set_current_direction(self, directions_list_new, carla_direction):
        if carla_direction == 3.0:
            self._state.direction = -1
        if carla_direction == 4.0:
            self._state.direction = 1
        if carla_direction == 2.0 or len(directions_list_new) == 1:
            self._state.direction = list(directions_list_new)[0]
        if carla_direction == 5.0 or carla_direction == 0.0:
            self._state.direction = 0

        ### set the correct GT centerline maps
        # set the centerlines according to the decided direction
        # and to the current street type
        direction = self._state.direction
        is_straight = direction == 0 or directions_list_new == {0}
        is_c1 = (direction == -1 and directions_list_new == {0,-1}) or \
                (direction == 1 and directions_list_new == {1,-1}) or \
                directions_list_new == {-1} or directions_list_new == {1}
        is_c2 = (direction == 1 and directions_list_new == {0,1}) or \
                (direction ==-1 and directions_list_new == {1,-1})

        # set the centerlines accordingly
        if is_straight: self._centerlines.set_centerlines('straight')
        if is_c1: self._centerlines.set_centerlines('c1')
        if is_c2: self._centerlines.set_centerlines('c2')

<<<<<<< HEAD
    def _compute_action(self,carla_direction, direction):
        # Predict the intermediate representations
        prediction = self._neural_net.predict(self._state.image_hist, [direction])
=======
    def _compute_action(self,carla_direction, direction):    
        start = time.time()
        # Predict the intermediate representations
        prediction = self._neural_net.predict(self._state.image_hist, [direction])
        
        logging.info("Time for prediction: {}".format(time.time() - start))
>>>>>>> upstream/master
        logging.info("CARLA Direction {}, Real Direction {}".format(carla_direction, direction))

        # update the speed limit if a speed limit sign is detected
        if prediction['speed_sign'][0] != -1:
<<<<<<< HEAD
            self._state.speed_limit = prediction['speed_sign'][0]
=======
            self._state.speed_limit = prediction['speed_sign']
>>>>>>> upstream/master

        # Calculate the control
        control = Control()
        control.throttle, control.brake = self._longitudinal_control(prediction, direction)
        control.steer = self._lateral_control(prediction)

        return control

    def _longitudinal_control(self, prediction, direction) :
        """
        calculate the _longitudinal_control
        the constants (c, d, curve_slowdown) are defined on top of the file
        """
        ### Variable setup
        throttle = 0
        brake = 0

<<<<<<< HEAD
        # get the state variables
        speed = self._state.speed
        limit = self._state.speed_limit

        # uncomment next line for similar conditions to carla paper
=======
        # unpack the state variables
        speed = self._state.speed
        limit = self._state.speed_limit

        # uncomment for similar conditions to carla paper
>>>>>>> upstream/master
        limit = 30

        # determine whether to use other states than cruising
        cruising_only = not self._agents_present

<<<<<<< HEAD
        # brake when driving into a curve]
=======
        # brake when driving into a curve
>>>>>>> upstream/master
        if direction: limit -= self.curve_slowdown

        # get the distance to the leading car
        veh_distance = prediction['veh_distance']
        # half of the speed indicator
        is_following = veh_distance < np.clip(limit, 30, 50)
        # optimal car following model
        following_speed = limit * (1-np.exp(-self.c/limit*veh_distance-self.d))

        ### State machine
<<<<<<< HEAD
        if prediction['hazard_stop'][0] and prediction['hazard_stop'][1] > 0.8 and self._agents_present:
            state_name  = 'hazard_stop'
            prediction_proba = prediction['hazard_stop'][1]
            brake = 1.0

        elif prediction['red_light'][0] and prediction['red_light'][1] > 0.98 and self._agents_present:
            state_name = 'red_light'
            prediction_proba = prediction['red_light'][1]
            throttle = 0
            # brake depended on speed
            if speed > 5: brake = 0.8*(speed/30.0)
            else: brake = 1.0
=======
        if prediction['hazard_stop'][0] \
        and prediction['hazard_stop'][1] > 0.9 \
        and self._agents_present:
            state_name  = 'hazard_stop'
            prediction_proba = prediction['hazard_stop'][1]
            brake = 1

        elif prediction['red_light'][0] \
        and prediction['red_light'][1] > 0.98 \
        and self._agents_present:
            state_name = 'red_light'
            prediction_proba = prediction['red_light'][1]
            throttle = 0
            if speed > 5: 
                # brake if driving to fast
                brake = 0.8*(speed/30.0)
            else: 
                # fully brake if close to standing still            
                brake = 1.0
>>>>>>> upstream/master

        elif is_following and self._agents_present:
            state_name = 'following'
            prediction_proba = 1.0
            desired_speed = following_speed
            self._PID_follow.update(desired_speed - speed)
            throttle = -self._PID_follow.output

        else: # is cruising
            state_name = 'cruising'
            prediction_proba = 1.0
            desired_speed = limit
            self._PID_cruise.update(desired_speed - speed)
            throttle = -self._PID_cruise.output

        logging.info('STATE: {}, PROBA: {:.4f} %'.format(state_name, prediction_proba*100))
<<<<<<< HEAD
        logging.info('SPEED LIMIT: {}, SPEED {:.2f}'.format(limit, speed))
=======
>>>>>>> upstream/master

        ### Additional Corrections
        # slow down speed limit is exceeded
        if speed > limit + 10: brake = 0.3 * (speed/30)

        # clipping
        throttle = np.clip(throttle, 0, 0.8)
        brake = np.clip(brake, 0, 1)
        if brake: throttle=0

        return throttle, brake

    def _lateral_control(self, prediction):
        """
        function implements the lateral control algorithm
        input:
<<<<<<< HEAD
            - vehicle speed
            - front axle position
            - vehicle yaw
            - distance to closest pixel on center line [with correct sign]
            - yaw in closest pixel on center line
            output:
=======
        - vehicle speed
        - front axle position
        - vehicle yaw
        - distance to closest pixel on center line [with correct sign]
        - yaw in closest pixel on center line
        output:
>>>>>>> upstream/master
        - delta signal in [-1,1]
        """
        # vehicle state
        v = self._state.speed

        # when standing don't steer
        if abs(v)<=0.1: return 0

        # choose value for k and d depending on the street
        if self._state.direction == 0:
            k = self.K0_STANLEY
            d = self.DELTA0
        elif self._state.direction ==-1:
            k = self.Kl_STANLEY
            d = self.DELTAl
        elif self._state.direction ==1:
            k = self.Kr_STANLEY
            d = self.DELTAr

        # stanley control
        theta_e = prediction['relative_angle']
        theta_d = math.atan2(k * prediction['center_distance'], v)
        delta = theta_e + theta_d

        # normalize delta
        delta /= MAX_STEER

        # get delta sign, damping is calculed using the absolute delta
        delta_sign = np.sign(delta)
        delta = abs(delta)

        # add exponential damping
        decay = d * math.exp(-self.EXP_DECAY*delta)
        logging.info("DECAY: {}".format(decay))
        delta -= decay
        delta = np.clip(delta, 0, 1)

        # return the signed delta
        return delta_sign*delta

    def _get_location_and_orientation(self, measurements):
        # get the location
        location_world = [measurements.player_measurements.transform.location.x,
                          measurements.player_measurements.transform.location.y,
                          measurements.player_measurements.transform.location.z]
        location_map = self._map.convert_to_pixel(location_world)

        # get the orientation
        veh_ori_x = measurements.player_measurements.transform.orientation.x,
        veh_ori_y = measurements.player_measurements.transform.orientation.y,
        psi = math.atan2(veh_ori_y[0],veh_ori_x[0]) # angle in the world COS

        return location_map[:2], psi

    def _get_front_axle(self, location, psi):
        # calculate the position of the front axle
        point = self._vehicle_to_world_COS(location, (0, 8.7644), psi)

        return (int(point[0]), int(point[1]))

    def _vehicle_to_world_COS(self, origin, point, psi):
        """
        transform a 2d point from the vehicle COS to the world COS
        """
        x_new = origin[0] - point[0]*math.sin(psi) + point[1]*math.cos(psi)
        y_new = origin[1] + point[0]*math.cos(psi) + point[1]*math.sin(psi)

        return (x_new, y_new)

    def _get_accel(self, measurements, psi):
        # get the absolute aceleration in the cars COS
        acceleration = measurements.player_measurements.acceleration
        a_x, a_y = acceleration.x, acceleration.y

        # calculate in the relative COS
        a_x_rel = a_x*math.cos(psi) + a_y*math.sin(psi)
        a_y_rel = -a_x*math.sin(psi) + a_y*math.cos(psi)

        return a_x_rel, a_y_rel

    def get_GT(self):
        """"
        This functions returns the current distance to the center line
        and the current directions_list_new

        """
        d = {}
        d['center_distance'] = self._state.center_distance_GT
        d['direction'] = self._state.direction
        d['long_accel'] = self._state.long_accel
        d['lat_accel'] = self._state.lat_accel

        return d
