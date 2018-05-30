from math import cos

class Model:
    def __init__(self, position_bounds, velocity_bounds):
        self.position_bounds = position_bounds
        self.velocity_bounds = velocity_bounds

        self.reset_state()

    def execute_action(self, acceleration):
        self.state = self._generate_next_state(acceleration)
        return self.get_current_state()
    
    def get_current_state(self):
        return self.state

    def get_reward(self):
        if(self.state["position"] == self.position_bounds[1]):
            return 10.0
        else:
            return -1.0

    def is_final_state(self):
        return (self.state["position"] in self.position_bounds)

    def is_success_state(self):
        return (self.state["position"] == self.position_bounds[1])

    def reset_state(self):
        self.state = self._generate_state(0.0, 0.0)

    def _check_bounds(self, value, bounds):
        if(value <= bounds[0]):
            return bounds[0]
        elif(value >= bounds[1]):
            return bounds[1]
        else:
            return value
    
    def _generate_next_state(self, acceleration):
        next_velocity = self._calc_velocity(self.state, acceleration)
        next_position = self._calc_position(self.state, next_velocity)
        return self._generate_state(next_position, next_velocity)

    def _calc_velocity(self, state, acceleration):
        if(state["position"] in self.position_bounds):
            return 0

        next_velocity = state["velocity"] + (0.001 * acceleration) - (0.0025 * cos(3 * state["position"]))
        next_velocity = self._check_bounds(next_velocity, self.velocity_bounds)
        return next_velocity

    def _calc_position(self, state, next_velocity):
        next_position = state["position"] + next_velocity
        next_position = self._check_bounds(next_position, self.position_bounds)
        return next_position

    def _generate_state(self, position, velocity):
        return {
            "position": position,
            "velocity": velocity
        }