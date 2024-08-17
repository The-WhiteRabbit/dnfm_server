import time
from typing import Tuple
import json
from scrcpy_adb import ScrcpyADB
import math
import random

class GameControl:
    def __init__(self, adb: ScrcpyADB,config):
        with open(config, 'r') as file:
            self.config = json.load(file)
        self.adb = adb
        self.move_touch = "none"
        self.attack_touch = "none"
        self.pos = [0,0]
        self.last_move = [self.config['joystick']['center'][0],self.config['joystick']['center'][1]]
    def calc_mov_point(self, angle: int) -> Tuple[int, int]:
        rx, ry = (int(self.config['joystick']['center'][0]), int(self.config['joystick']['center'][1]))
        if angle == 0:
            return rx, ry
        angle = angle%360
        if angle < 0:
            angle = 360 + angle
        r = int(self.config['joystick']['radius'])
        x = rx + r * math.cos(angle * math.pi / 180)
        y = ry - r * math.sin(angle * math.pi / 180)
        return int(x), int(y)
    def move(self, angle: int):
        # 计算轮盘x, y坐标
        x, y = self.calc_mov_point(angle)
        # x, y  = self._ramdon_xy(x, y)
        if angle == 0:
            if self.move_touch == "none":
                return
            self.move_touch = "none"
            self.adb.touch_up(self.last_move[0], self.last_move[1],1)
            return
        else:
            if self.move_touch == "none":
                self.move_touch = "start"
                self.adb.touch_down(x, y,1)
                self.last_move = [x,y]
            else:
                self.move_touch = "move"
                self.adb.touch_move(x, y,1)
                # self.adb.touch_swipe(self.last_move[0],self.last_move[1],x,y,5,0.001)
                self.last_move = [x,y]
    def attack(self,flag : bool=True):
        if flag == False:
            if self.attack_touch == "none":
                return
            else:
                self.attack_touch = "none"
                x, y = (int(self.config['attack'][0]), int(self.config['attack'][1]))
                x, y = self._ramdon_xy(x, y)
                self.adb.touch_move(x, y,2)
                self.adb.touch_up(x, y,2)
                return 
        x, y = (int(self.config['attack'][0]), int(self.config['attack'][1]))
        x, y = self._ramdon_xy(x, y)
        self.pos = [x,y]
        if self.attack_touch == "none":
            self.attack_touch = "start"
            self.adb.touch_down(x, y,2)
        else:
            self.adb.touch_move(x, y,2)
    def skill(self, name:str, t: float = 0.01):
        if isinstance(self.config[name], str):
            self.Roulette(name)
            return
        if self.attack_touch != "none":
            self.adb.touch_up(self.pos[0], self.pos[1],2)
            self.attack_touch == "none"
        x, y = (int(self.config[name][0]), int(self.config[name][1]))
        x, y = self._ramdon_xy(x, y)
        self.adb.touch_down(x, y,3)
        time.sleep(t)
        self.adb.touch_up(x, y,3)
    def Roulette(self, name:str):#轮盘技能
        dir = {"left":[-100,0],"right":[100,0],"up":[0,-100],"down":[0,100]}
        start_x = self.config["Roulette"][0]
        start_y = self.config["Roulette"][1]
        end_x = self.config["Roulette"][0] + dir[self.config[name]][0]
        end_y = self.config["Roulette"][1] + dir[self.config[name]][1]
        start_x,start_y = self._ramdon_xy(start_x,start_y)
        end_x,end_y = self._ramdon_xy(end_x,end_y)
        self.adb.touch_swipe(start_x,start_y,end_x,end_y)
    def jump(self):
        if self.attack_touch != "none":
            self.adb.touch_up(self.pos[0], self.pos[1],2)
            self.attack_touch == "none"
        x, y = (int(self.config['Jump'][0]), int(self.config['Jump'][1]))
        x, y = self._ramdon_xy(x, y)
        self.adb.touch_down(x, y,3)
        time.sleep(0.05)
        self.adb.touch_up(x, y,3)
    def back_jump(self):
        if self.attack_touch != "none":
            self.adb.touch_up(self.pos[0], self.pos[1],2)
            self.attack_touch == "none"
        x, y = (int(self.config['Jump_Back'][0]), int(self.config['Jump_Back'][1]))
        x, y = self._ramdon_xy(x, y)
        self.adb.touch_down(x, y,3)
        time.sleep(0.05)
        self.adb.touch_up(x, y,3)
    def flash(self, angle: float):
        self.move(angle)
        if self.attack_touch != "none":
            self.adb.touch_up(self.pos[0], self.pos[1],2)
            self.attack_touch == "none"
        x, y = (int(self.config['Jump_Back'][0]), int(self.config['Jump_Back'][1]))
        x, y = self._ramdon_xy(x, y)
        time.sleep(0.05)
        self.adb.touch_down(x, y,2)
        time.sleep(0.05)
        self.adb.touch_up(x, y,2)
        time.sleep(0.05)
        self.adb.touch_down(x, y,2)
        time.sleep(0.05)
        self.adb.touch_up(x, y,2)
    def click(self, x, y, t=0.1):
        self.reset()
        time.sleep(0.15)
        self.adb.touch_down(x, y)
        time.sleep(t)
        self.adb.touch_up(x, y)
    def reset(self):
        self.move(0)
        self.attack(False)
    def _ramdon_xy(self, x, y):
        x = x + random.randint(-5, 5)
        y = y + random.randint(-5, 5)
        return x, y

