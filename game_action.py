from typing import Tuple
from game_control import GameControl
from scrcpy_adb import ScrcpyADB
import time
import cv2
import math
import numpy as np
from collections import deque
import threading

def calculate_center(box):# 计算矩形框的底边中心点坐标
    return ((box[0] + box[2]) / 2, box[3])
def calculate_distance(center1, center2):# 计算两个底边中心点之间的欧几里得距离
    return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
def find_closest_box(boxes, target_box):# 计算目标框的中心点
    target_center = calculate_center(target_box)# 初始化最小距离和最近的box
    min_distance = float('inf')
    closest_box = None# 遍历所有box，找出最近的box
    for box in boxes:
        center = calculate_center(box)
        distance = calculate_distance(center, target_center) 
        if distance < min_distance:
            min_distance = distance
            closest_box = box
    return closest_box,min_distance
def find_farthest_box(boxes, target_box):
    target_center = calculate_center(target_box)# 计算目标框的中心点
    max_distance = -float('inf')# 初始化最大距离和最远的box
    farthest_box = None
    for box in boxes:# 遍历所有box，找出最远的box
        center = calculate_center(box)
        distance = calculate_distance(center, target_center)
        if distance > max_distance:
            max_distance = distance
            farthest_box = box
    return farthest_box, max_distance
def find_closest_or_second_closest_box(boxes, point):
    """找到离目标框最近的框或第二近的框"""
    if len(boxes) < 2:
        # 如果框的数量少于两个，直接返回最近的框
        target_center = point
        closest_box = None
        min_distance = float('inf')
        for box in boxes:
            center = calculate_center(box)
            distance = calculate_distance(center, target_center)
            if distance < min_distance:
                min_distance = distance
                closest_box = box
        return closest_box,distance
    # 如果框的数量不少于两个
    target_center = point
    # 初始化最小距离和最近的框
    min_distance_1 = float('inf')
    closest_box_1 = None
    # 初始化第二近的框
    min_distance_2 = float('inf')
    closest_box_2 = None
    for box in boxes:
        center = calculate_center(box)
        distance = calculate_distance(center, target_center)
        if distance < min_distance_1:
            # 更新第二近的框
            min_distance_2 = min_distance_1
            closest_box_2 = closest_box_1
            # 更新最近的框
            min_distance_1 = distance
            closest_box_1 = box
        elif distance < min_distance_2:
            # 更新第二近的框
            min_distance_2 = distance
            closest_box_2 = box
    # 返回第二近的框
    return closest_box_2,min_distance_2
def find_close_point_to_box(boxes, point):#找距离点最近的框
    target_center = point# 初始化最小距离和最近的box
    min_distance = float('inf')
    closest_box = None# 遍历所有box，找出最近的box
    for box in boxes:
        center = calculate_center(box)
        distance = calculate_distance(center, target_center) 
        if distance < min_distance:
            min_distance = distance
            closest_box = box
    return closest_box,min_distance
def calculate_point_to_box_angle(point, box):#计算点到框的角度
    center1 = point
    center2 = calculate_center(box)
    delta_x = center2[0] - center1[0]# 计算相对角度（以水平轴为基准）
    delta_y = center2[1] - center1[1]
    angle = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle)# 将角度转换为度数
    adjusted_angle = - angle_degrees
    return adjusted_angle
def calculate_angle(box1, box2): 
    center1 = calculate_center(box1)
    center2 = calculate_center(box2)
    delta_x = center2[0] - center1[0]# 计算相对角度（以水平轴为基准）
    delta_y = center2[1] - center1[1]
    angle = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle)# 将角度转换为度数
    adjusted_angle = - angle_degrees
    return adjusted_angle
def calculate_gate_angle(point, gate): 
    center1 = point
    center2 = ((gate[0]+gate[2])/2,(gate[3]-gate[1])*0.65+gate[1])
    delta_x = center2[0] - center1[0]# 计算相对角度（以水平轴为基准）
    delta_y = center2[1] - center1[1]
    angle = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle)# 将角度转换为度数
    adjusted_angle = - angle_degrees
    return adjusted_angle

def calculate_angle_to_box(point1,point2):#计算点到点的角度
    angle = math.atan2(point2[1] -point1[1], point2[0] - point1[0])# 计算从点 (x, y) 到中心点的角度
    angle_degrees = math.degrees(angle)# 将角度转换为度数
    adjusted_angle = - angle_degrees
    return adjusted_angle
def calculate_iou(box1, box2):
    # 计算相交区域的坐标
    inter_x_min = max(box1[0], box2[0])
    inter_y_min = max(box1[1], box2[1])
    inter_x_max = min(box1[2], box2[2])
    inter_y_max = min(box1[3], box2[3])
    # 计算相交区域的面积
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    # 计算每个矩形的面积和并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    # 计算并返回IoU
    return inter_area / union_area if union_area > 0 else 0
def normalize_angle(angle):# 将角度规范化到 [-180, 180) 的范围内
    angle = angle % 360
    if angle >= 180:
        angle -= 360
    return angle
def are_angles_on_same_side_of_y(angle1, angle2):# 规范化角度
    norm_angle1 = normalize_angle(angle1)
    norm_angle2 = normalize_angle(angle2)# 检查是否在 y 轴的同侧
    return (norm_angle1 >= 0 and norm_angle2 >= 0) or (norm_angle1 < 0 and norm_angle2 < 0)
def is_image_almost_black(image, threshold=30):# 读取图片
    image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)# 检查图片是否成功读取
    num_black_pixels = np.sum(image < threshold)
    total_pixels = image.size
    black_pixel_ratio = num_black_pixels / total_pixels# 定义一个比例阈值来判断图片是否接近黑色
    return black_pixel_ratio > 0.7        
names =['Monster', #0
        'Monster_ds', #1
        'Monster_szt', #2
        'card', #3
        'equipment',#4 
        'go', #5
        'hero', #6
        'map', #7
        'opendoor_d', #8
        'opendoor_l', #9
        'opendoor_r', #10
        'opendoor_u', #11
        'pet',# 12
        'Diamond']# 13
from hero.naima import Naima
class GameAction:
    def __init__(self, ctrl: GameControl,queue):
        self.queue = queue
        self.ctrl = ctrl
        self.detect_retry = False
        self.pre_state = True
        self.stop_event = True
        self.reset_event = False
        self.control_attack = Naima(ctrl)
        self.room_num = -1
        self.buwanjia = [8,10,10,11,9,10,10,10,10,10,10,10,10,10]
        self.thread_run = True
        self.thread = threading.Thread(target=self.control)  # 创建线程，并指定目标函数
        self.thread.daemon = True  # 设置为守护线程（可选）
        self.thread.start()
        
    def reset(self):
        self.thread_run = False
        time.sleep(0.1)
        self.room_num = -1
        self.detect_retry = False
        self.pre_state = True
        self.thread_run = True
        self.thread = threading.Thread(target=self.control)  # 创建线程，并指定目标函数
        self.thread.daemon = True  # 设置为守护线程（可选）
        self.thread.start()
    def control(self):
        last_room_pos = []
        hero_track = deque()
        hero_track.appendleft([0,0])
        last_angle = 0
        while self.thread_run:
            if self.stop_event:
                time.sleep(0.001)
                self.ctrl.reset()
                continue
            if self.queue.empty():
                time.sleep(0.001)
                continue
            image,boxs = self.queue.get()
            if is_image_almost_black(image):
                if self.pre_state == False:
                    print("过图")
                    last_room_pos = hero_track[0]
                    hero_track = deque()
                    hero_track.appendleft([1-last_room_pos[0],1-last_room_pos[1]])
                    last_angle = 0
                    self.ctrl.reset()
                    self.pre_state = True
                else:continue
            hero = boxs[boxs[:,5]==6][:,:4]
            gate = boxs[boxs[:,5]==self.buwanjia[self.room_num]][:,:4]
            arrow = boxs[boxs[:, 5] == 5][:,:4]
            equipment = [[detection[0], detection[1] + (detection[3] - detection[1]), detection[2], detection[3] + (detection[3] - detection[1]), detection[4], detection[5]]
                        for detection in boxs if detection[5] == 4 and detection[4] > 0.3]
            monster = boxs[boxs[:,5]<=2][:,:4]
            card = boxs[boxs[:,5]==3][:,:4]
            Diamond = boxs[boxs[:,5]==13][:,:4]
            angle = 0
            outprint = ''
            if self.pre_state == True :
                if len(hero) > 0:#记录房间号
                    self.room_num += 1
                    self.pre_state = False
                    print("房间号：",self.room_num)
                    print("目标",self.buwanjia[self.room_num])
                else:
                    continue
            self.calculate_hero_pos(hero_track,hero)#计算英雄位置
            if len(card)>=8:
                time.sleep(1)
                self.ctrl.click(0.25*image.shape[0],0.25*image.shape[1])
                self.detect_retry = True
                time.sleep(2.5)
            if len(monster)>0:
                outprint = '有怪物'
                angle = self.control_attack.control(hero_track[0],image,boxs,self.room_num)
            elif len(equipment)>0:
                outprint = '有材料'
                if len(gate)>0:
                    close_gate,distance = find_close_point_to_box(gate,hero_track[0])
                    farthest_item,distance = find_farthest_box(equipment,close_gate)
                    angle = calculate_point_to_box_angle(hero_track[0],farthest_item)
                else:
                    close_item,distance = find_close_point_to_box(equipment,hero_track[0])
                    angle = calculate_point_to_box_angle(hero_track[0],close_item)
                self.ctrl.attack(False)
                self.ctrl.move(angle)
            elif len(gate)>0:
                outprint = '有门'
                if self.buwanjia[self.room_num] == 9:#左门
                    close_gate,distance = find_close_point_to_box(gate,hero_track[0])
                    angle = calculate_gate_angle(hero_track[0],close_gate)
                    self.ctrl.attack(False)
                    self.ctrl.move(angle)
                else:
                    close_gate,distance = find_close_point_to_box(gate,hero_track[0])
                    angle = calculate_point_to_box_angle(hero_track[0],close_gate)
                    self.ctrl.attack(False)
                    self.ctrl.move(angle)
            elif len(arrow)>0 and self.room_num != 4:
                outprint = '有箭头'
                close_arrow,distance = find_closest_or_second_closest_box(arrow,hero_track[0])
                angle = calculate_point_to_box_angle(hero_track[0],close_arrow)
                self.ctrl.move(angle)
                self.ctrl.attack(False)
            elif self.detect_retry == True:
                #重新挑战：自行发挥
                print("detect_retry")
                self.ctrl.move(0)
                self.detect_retry =False
                self.room_num = 0
                hero_track = deque()
                hero_track.appendleft([0,0])
            else :
                outprint = "无目标"
                if self.room_num == 4:
                    angle = calculate_angle_to_box(hero_track[0],[0.25,0.6])
                else:
                    angle = calculate_angle_to_box(hero_track[0],[0.5,0.75])
                self.ctrl.move(angle)
                self.ctrl.attack(False)
            print(f"\r当前进度:{outprint},角度{angle}，位置{hero_track[0]}", end="")
    def calculate_hero_pos(self,hero_track,boxs):
        if len(boxs)==0:
            None
        elif len(boxs)==1:
            hero_track.appendleft(calculate_center(boxs[0]))
        elif len(boxs)>1:
            for box in boxs:
                if calculate_distance(box,hero_track[0])<0.1:
                    hero_track.appendleft(box)
                    return
                hero_track.appendleft(hero_track[0])