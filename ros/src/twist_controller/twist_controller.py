from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy #for get_time()

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius,
                 wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        """ TODO: The car is wandering and might need to get fixed """
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # Values for PID controller (empirically chosen)
        kp = 0.2
        ki = 0.01
        kd = 3.
        mn = 0.     #Minimum throttle value
        mx = 0.2    #Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        # High-frequency noise
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit # comfort values, jerk etc
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time = rospy.get_time()
        
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        """
        Parameters
        ----------
        current_vel : float
            current velocity
        dbw_enabled : bool
            Is drive-by-wire enabled? While stopping, it might be 
            turned off to avoid the accumulation of error 
            (stopping versus predicted driving)
        linear_vel : float
            linear velocity component (x)
        angular_vel : float
            angular velocity component (around z-axis)
        Returns
        -------
        float, float, float
            throttle, brake, steering
        """
        if not dbw_enabled:
            # Safety driver took over
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        # Velocity low pass filter to remove high-frequency noise
        #current_vel = self.vel_lpf(current_vel) #from walkthrough
        #current_vel = self.vel_lpf.filt(current_vel)
        current_vel = current_vel
        
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        # Get the step size of the PID controller
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        # Should the car accelerate or brake or do nothing?
        throttle = self.throttle_controller.step(vel_error, sample_time)
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 700 #N*m - to hold the car in place if we stop at a traffic light
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m # has to be positive
        else:
            brake = 0
        return throttle, brake, steering
