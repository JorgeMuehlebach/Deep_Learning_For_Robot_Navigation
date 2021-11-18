from numpy.lib.function_base import average
from stable_baselines.common.callbacks import BaseCallback

from pynput import keyboard
import time
import numpy as np
class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """
    def __init__(self, verbose=0, stopping_letter='A', logging=True, using_ppo = False):
        """

        Args:
            verbose (int, optional): see super. Defaults to 0.
            stopping_letter (str, optional): pressing this character will stop training and save the model
            . Defaults to 'A'.
            logging (bool, optional): set to false to stop logging. Useful if you are using an gym env that doesn't 
            record data in the same way the that husky_env.py does. It means that the stopping letter will still work
            using_ppo (bool, optional): weather or not ppo is being used (affects logging). Defaults to False.
        """
        super(CustomCallback, self).__init__(verbose)
        self.using_ppo = using_ppo
        self.stopping_letter = stopping_letter
        # used for the stopping letter press
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.abort_training = False
        self.listener.start()
        self.episode_counter = 0
        self.successes = 1
        self.failures = 1
        self.previous_sucess_ratio = -1
        self.start_time = time.time()
        self.total_steps = 0
        self.outcomes = []
        self.moving_success_rate = []
        self.moving_avg_episode_reward = []
        self.start_time = time.time()
        self.stats_each_episode = []
        self.episode_smoothness_rewards = []
        self.smoothness_reward_this_episode = 0
        self.logging = logging
        
    def on_press(self, key):
        try:
            k = key.char  # single-char keys
        except:
            k = key.name  # other keys
        if k == self.stopping_letter:
            for i in range(10):
                print(key)
            self.abort_training = True

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.logging == True:
            smoothness_reward = 0
            fail = True
            done = True
            episode_counter = 0
            episode_rewards = 0
            ingoal = False
            if self.using_ppo:
                smoothness_reward = self.training_env.get_attr('smoothness_reward')[0]
                fail = self.training_env.get_attr('fail')[0]
                done = self.training_env.get_attr('done')[0]
                episode_counter = self.training_env.get_attr('episode_counter')[0]
                episode_rewards = self.training_env.get_attr('REWARD_PER_EPISODE')[0]
                ingoal = self.training_env.get_attr('ingoal')[0]
            else:
                smoothness_reward = self.training_env.smoothness_reward
                fail = self.training_env.fail
                done = self.training_env.done
                episode_counter = self.training_env.episode_counter
                episode_rewards = self.training_env.REWARD_PER_EPISODE
                ingoal = self.training_env.ingoal
            

            self.smoothness_reward_this_episode += smoothness_reward
            if ingoal:
                self.outcomes.append(1)
            elif fail:

                self.outcomes.append(0)
            # end of the episode
            if done:
                print('episodes:', episode_counter)
                print('stopping character: ', self.stopping_letter)
                self.episode_smoothness_rewards.append(self.smoothness_reward_this_episode)
                self.smoothness_reward_this_episode = 0

                self.episode_counter += 1
                #episode_rewards = self.training_env.get_attr('REWARD_PER_EPISODE')[0]
                episode_stats = []
                episode_stats.append(self.episode_counter)
                episode_stats.append(time.time() - self.start_time)
                episode_stats.append(episode_rewards[-1])
                # prints most recent episode reward
                print("this episodes reward: ", episode_rewards[-1])
                n = len(episode_rewards)
                moving_avg_N = 100
                if n <= moving_avg_N:
                    moving_avg = np.mean(episode_rewards)
                    self.moving_avg_episode_reward.append(moving_avg)
                    print("avg episode reward: ", moving_avg)
                    episode_stats.append(moving_avg)
                    moving_smoothness = np.mean(self.episode_smoothness_rewards)
                    episode_stats.append(moving_smoothness)
                    episode_stats.append(self.episode_smoothness_rewards[-1])
                    
                else:
                    moving_avg = np.mean(episode_rewards[-moving_avg_N:])
                    self.moving_avg_episode_reward.append(moving_avg)
                    print("avg episode reward: ", moving_avg)
                    episode_stats.append(moving_avg)
                    moving_smoothness = np.mean(self.episode_smoothness_rewards[-moving_avg_N:])
                    episode_stats.append(moving_smoothness)
                    episode_stats.append(self.episode_smoothness_rewards[-1])

                
                outcomes_n = len(self.outcomes)
                if outcomes_n <= moving_avg_N:
                    moving_avg = np.mean(self.outcomes)
                    self.moving_avg_episode_reward.append(moving_avg)
                    print("success rate: ", moving_avg)
                    episode_stats.append(moving_avg)
                else:
                    moving_avg = np.mean(self.outcomes[-moving_avg_N:])
                    self.moving_avg_episode_reward.append(moving_avg)
                    print("sucess rate: ", moving_avg)
                    episode_stats.append(moving_avg)
            
                self.stats_each_episode.append(episode_stats)

        if self.abort_training:
            return False

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
