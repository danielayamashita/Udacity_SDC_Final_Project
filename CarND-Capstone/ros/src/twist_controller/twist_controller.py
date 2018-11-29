from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy 

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # In my PID project I chose KP=0.2, KI=0.004, KD = 3, the parameters suggested by Sebatian in Term2, video 16/11 (PID implemnentation)
        kp = 0.2    #0.2
        ki = 3.  #3.
        kd = 0.004     #0.004
        mn = 0.     # min throttle value
        mx = 0.2    # max throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        tau = 0.5 # 1/(2*pi*tau) == cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_vel = 0 #?
        
        self.last_time = rospy.get_time()

    def control(self, current_velocity, dbw_enabled, linear_velocity, angular_velocity):
        # Returns throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        current_velocity = self.vel_lpf.filt(current_velocity)
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)
        
        vel_error = linear_velocity - current_velocity
        self.last_vel = current_velocity
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0
        
        # TODO: is exact floating point comparison sufficient?
        if linear_velocity == 0. and current_velocity < 0.1:
            rospy.logerr("TC::control: Setting throttle to 0 and brake to 400")
            throttle = 0.
            brake = 400 # N*m  - to hold the car in place if we are stopped at a light. Acceleration  ~ 1m/s^2
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m
            rospy.logwarn("TC::control: Setting throttle to 0 and brake to %.5f", brake)
        
        return throttle, brake, steering
        