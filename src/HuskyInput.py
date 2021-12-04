
from pynput import keyboard

class HuskyInputController:
     def __init__(self, env) -> None:
        self.listener = keyboard.Listener(
        on_press=self.on_press,
        on_release=self.on_release)
        self.listener.start()
        self.huskyEnv = env
    
     def on_press(self, key):
        if key == keyboard.Key.esc:
            return False  # stop listener
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys
        
        if k == 'up':
            self.huskyEnv.set_wheels_left_speed(1)
            self.huskyEnv.set_wheels_right_speed(1)
        elif k=='left':
            self.huskyEnv.set_wheels_right_speed(1)
        elif k=='right':
            self.huskyEnv.set_wheels_left_speed(1)

     def on_release(self, key):
        k = key.name
        if k == 'up':
            self.huskyEnv.set_wheels_left_speed(0)
            self.huskyEnv.set_wheels_right_speed(0)
        elif k=='left':
            self.huskyEnv.set_wheels_right_speed(0)
        elif k=='right':
            self.huskyEnv.set_wheels_left_speed(0)

    
     def stop(self):
        self.listener.join() 
