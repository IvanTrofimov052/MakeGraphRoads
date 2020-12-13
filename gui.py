from calculated import *

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from PIL import Image


# this class working with GUI
class GUI:
    calculated = Calculated()

    def game_loop(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--env-name', default=None)
        parser.add_argument('--map-name', default='udem1')
        parser.add_argument('--distortion', default=False, action='store_true')
        parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
        parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
        parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
        parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
        parser.add_argument('--seed', default=1, type=int, help='seed')
        args = parser.parse_args()

        if args.env_name and args.env_name.find('Duckietown') != -1:
            env = DuckietownEnv(
                seed=args.seed,
                map_name=args.map_name,
                draw_curve=args.draw_curve,
                draw_bbox=args.draw_bbox,
                domain_rand=args.domain_rand,
                frame_skip=args.frame_skip,
                distortion=args.distortion,
            )
        else:
            env = gym.make(args.env_name)

        env.reset()
        env.render()

        # Register a keyboard handler
        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)

        def update(dt):
            """
            This function is called at every frame to handle
            movement/stepping and redrawing
            """

            lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
            distance_to_road_center = lane_pose.dist
            print(lane_pose)

            action = np.array([0.0, 0.0])

            if key_handler[key.UP]:
                action = np.array([0.44, 0.0])
            if key_handler[key.DOWN]:
                action = np.array([-0.44, 0])
            if key_handler[key.LEFT]:
                action = np.array([0.35, +1])
            if key_handler[key.RIGHT]:
                action = np.array([0.35, -1])
            if key_handler[key.SPACE]:
                action = np.array([0, 0])

            # Speed boost
            if key_handler[key.LSHIFT]:
                action *= 1.5

            obs, reward, done, info = env.step(action)

            im = Image.fromarray(obs)

            if done:
                print('done!')
                env.reset()
                env.render()

            self.calculated.converter.analayze_image(im)

            env.render()

        pyglet.clock.schedule_interval(update, 5.0 / env.unwrapped.frame_rate)

        # Enter main event loop
        pyglet.app.run()

        env.close()