import sys
import keyboard
import cv2
import os
import traceback
from sys import platform
import argparse
import time
import math
import numpy as np
import tensorflow as tf
import scipy.spatial as spatial
import serial
from multiprocessing import Process, Manager, Queue, Event
from multiprocessing.connection import Client
import statistics
import subprocess
from models_and_utils import *
import imutils


############################# GLOBAL VARIABLES ######################################
command_port1 = 'COM7'
data_port1 = 'COM6'
command_port2 = 'COM3'
data_port2 ='COM12'
command_port3 = 'COM11'
data_port3 = 'COM4'
command_port4 = 'COM15'
data_port4 = 'COM16'

################## Load camera matrix and distortion coefficients function ##########################
def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

##############################RADAR SESSION CLASS#####################################

class RadarSession:

    CONFIG_FILE = 'mmw_pplcount_demo_default.cfg'
    TIMEOUT = 0.05  # TODO RESET TO 0.05
    POINT_CLOUD_TLV = 6
    TARGET_OBJECT_TLV = 7
    TARGET_INDEX_TLV = 8
    MAX_BUFFER_SIZE = 2 ** 15
    TLV_HEADER_BYTES = 8
    TARGET_LIST_TLV_BYTES = 68
    POINT_CLOUD_TLV_POINT_BYTES = 16
    MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]
    # word array to convert 4 bytes to a 32 bit number
    WORD = [1, 2 ** 8, 2 ** 16, 2 ** 24]

    def __init__(self, command_port, data_port):
        self.CLIport, self.Dataport = self.serialConfig(command_port, data_port)

    # Function to configure the serial ports and send the data from
    # the configuration file to the radar
    def serialConfig(self, command_port, data_port):
        # Open the serial ports for the configuration and the data ports
        CLIport = serial.Serial(command_port, 115200)
        Dataport = serial.Serial(data_port, 921600, timeout=self.TIMEOUT)
        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(self.CONFIG_FILE)]
        for i in config:
            CLIport.write((i + '\n').encode())
            time.sleep(0.01)
        return CLIport, Dataport

    # Funtion to read and parse the incoming data
    def readAndParseData(self):
        try:
            readBuffer = self.Dataport.read(
                    size=self.MAX_BUFFER_SIZE - 1)  # TODO inside brackets it was Dataport.in_waiting
            byteVec = np.frombuffer(readBuffer, dtype='uint8')
            byteBufferLength = len(byteVec)
            # Check for all possible locations of the magic word
            possibleLocs = np.where(byteVec == self.MAGIC_WORD[0])[0]
            # Confirm that it is the beginning of the magic word and store the index in startIdxes
            startIdxes = []
            for loc in possibleLocs:
                check = byteVec[loc:loc + 8]
                if np.all(np.array_equal(check, self.MAGIC_WORD)):
                    startIdxes.append(loc)
            for startIdx in startIdxes:
                # Read the total packet length
                totalPacketLen = np.matmul(byteVec[(startIdx + 20):(startIdx + 20 + 4)], self.WORD)
                # Check that all the packet has been read
                if (startIdx + totalPacketLen) <= byteBufferLength:
                    # Initialize the pointer index
                    idX = startIdx + 48
                    numTLVs = np.matmul(byteVec[idX:idX + 2], self.WORD[0:2])
                    idX += 4
                    # If the number of TLV messages is at most 3
                    if 0 < numTLVs <= 3:
                        # Read the TLV messages
                        for tlvIdx in range(numTLVs):
                            # Check the header of the TLV message
                            tlv_type = np.matmul(byteVec[idX:idX + 4], self.WORD)
                            idX += 4
                            # Read the data depending on the TLV message
                            if tlv_type == self.POINT_CLOUD_TLV:
                                tlv_length = np.matmul(byteVec[idX:idX + 4], self.WORD)
                                idX += (tlv_length - 4)
                            # Read the data depending on the TLV message
                            if tlv_type == self.TARGET_OBJECT_TLV:
                                tlv_length = np.matmul(byteVec[idX:idX + 4], self.WORD)
                                idX += 8  # +8 because i do not need the TID
                                number_of_targets = int((tlv_length - self.TLV_HEADER_BYTES) / self.TARGET_LIST_TLV_BYTES)

                                # Initialize array of 2D coordinates
                                xy = np.zeros((number_of_targets, 2), dtype=np.float32)
                                for target in range(number_of_targets):
                                    # Read the data for each target
                                    xy[target, 0] = byteVec[idX:idX + 4].view(dtype=np.float32)
                                    idX += 4
                                    xy[target, 1] = byteVec[idX:idX + 4].view(dtype=np.float32)
                                    idX += 64
                                return xy
            return None

        except:
            return None

############################## UTILITY FUNCTIONS #####################################

# This function opens a radar session at a specific command and data port, giving the radar session an id, and
# continuously waits on an event to capture data from the radar. The output of the radr is fed into a queue.
def radar_session(id, command_port, data_port, event, queue, *args):
    try:
        session = RadarSession(command_port, data_port)
        while True:
            event.wait()
            xy = session.readAndParseData()
            queue.put((id, xy))
            event.clear()
    except:
        print("Radar " + id + "session cannot be started")


# This function takes as input the 2D coordinates of the people detected by the three radars and converts them into the
# same reference frame, which is the reference frame of the first radar
def under_same_reference(coords1, coords2, coords3, coords4):
    # coords1 shape = (N,2) where N=people detected, 2=(x,y)
    # coords2 shape = (T,2) where T=people detected, 2=(x,y)
    # coords3 shape = (S,2) where S=people detected, 2=(x,y)
    if (coords1 is None) and (coords2 is None) and (coords3 is None) and (coords4 is None):
        return None
    radian2, radian3, radian4 = np.radians(-90), np.radians(-180), np.radians(90)
    if coords1 is None:
        coords1 = np.asarray([[np.nan, np.nan]])
    if coords2 is not None:
        coords2 = np.dot(coords2, np.asarray([[np.cos(radian2), -np.sin(radian2)], [np.sin(radian2), np.cos(radian2)]]))
    else:
        coords2 = np.asarray([[np.nan, np.nan]])
    if coords3 is not None:
        coords3 = np.dot(coords3, np.asarray([[np.cos(radian3), -np.sin(radian3)], [np.sin(radian3), np.cos(radian3)]]))
    else:
        coords3 = np.asarray([[np.nan, np.nan]])
    if coords4 is not None:
        coords4 = np.dot(coords4, np.asarray([[np.cos(radian4), -np.sin(radian4)], [np.sin(radian4), np.cos(radian4)]]))
    else:
        coords4 = np.asarray([[np.nan, np.nan]])

    output = np.concatenate((coords1, coords2, coords3, coords3, coords4), axis=0)
    return output[~np.isnan(output).any(axis=1)]


# This function matches camera- and radar-detected individuals together
def matching(pixel_keypoints, radar_data):
    # keypoint_angles shape = (N, 15, 2) where N=people detected, 15=human body keypoints, 2=(x-angle,y-angle)
    # radar_data shape = (N, 2) where N=people detected, 2=(x-angle, euclidean distance)
    # keypoint_mean_angles shape = (N,2) where N=people detected, 2=(x-angle, y-angle)

    keypoint_mean_angles = np.nanmean(pixel_keypoints, axis=1)
    keypoint_mean_angles = np.expand_dims(keypoint_mean_angles[:, 0], 1)

    # keypoint_mean_angles shape = (N,1) where N=people detected
    people_detected = keypoint_mean_angles.shape[0]
    queries = radar_data[:, 0].tolist()
    tree = spatial.KDTree(keypoint_mean_angles)
    indexes = []
    threshold = 200
    radar_data_indexes = []

    for radar_index, query in enumerate(queries):
        for p in range(people_detected):
            d, index = tree.query([query], k = p + 1)
            if isinstance(index, np.int32):
                if index not in indexes and d <= threshold:
                    indexes.append(index)
                    radar_data_indexes.append(radar_index)
                    break
            else:
                if index[-1] not in indexes and d[-1] <= threshold:
                    indexes.append(index[-1])
                    radar_data_indexes.append(radar_index)
                    break
    return indexes, radar_data_indexes

# If the pose is not occluded, the local and global coordinates of it are estimated using this function
def find_closest_point_on_line(line_point1, line_point2, target_point):
    line_vector = np.array([line_point2[0] - line_point1[0], line_point2[1] - line_point1[1], line_point2[2] - line_point1[2]])
    target_to_line = np.array([target_point[0] - line_point1[0], target_point[1] - line_point1[1], target_point[2] - line_point1[2]])
    t = np.dot(target_to_line, line_vector) / np.dot(line_vector, line_vector)
    closest_point = line_point1 + t * line_vector
    return closest_point
    
def calculate_shortest_distance(line_point1, line_point2, target_point):
    # Calculate the direction vector of the line
    line_vector = np.array([line_point2[0] - line_point1[0], line_point2[1] - line_point1[1], line_point2[2] - line_point1[2]])

    # Calculate the vector from the target point to any point on the line
    target_to_line = np.array([target_point[0] - line_point1[0], target_point[1] - line_point1[1], target_point[2] - line_point1[2]])

    # Calculate the cross product
    cross_product = np.cross(target_to_line, line_vector)

    # Calculate the distance
    distance = np.linalg.norm(cross_product) / np.linalg.norm(line_vector)
    return distance
    
def appendSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    if xyz.ndim == 1:
        xy = xyz[0]**2 + xyz[1]**2
        ptsnew[0] = np.sqrt(xy + xyz[2]**2)        #spherical
        ptsnew[1] = np.arctan2(np.sqrt(xy), xyz[2]) 
        ptsnew[2] = np.arctan2(xyz[1], xyz[0])
        """
        ptsnew[0] = np.sqrt(xy)                     #cylindrical
        ptsnew[1] = np.arctan2(xyz[1], xyz[0])
        ptsnew[2] = xyz[2]
        """
    else:
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)       #spherical
        ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) 
        ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])  
        """
        ptsnew[:,0] = np.sqrt(xy)                     #cylindrical
        ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) 
        ptsnew[:,2] = xyz[:,2]
        """
    return ptsnew   

def to_cartesian(spher):
    cart = np.zeros(spher.shape)
    if spher.ndim == 1:
        theta = spher[1]
        phi = spher[2]
        cart[0] = spher[0] * np.sin(theta) * np.cos(phi) #x  spherical
        cart[1] = spher[0] * np.sin(theta) * np.sin(phi)   #y
        cart[2] = spher[0] * np.cos(theta) #z
        """
        cart[0] = spher[0] * np.cos(theta)  #x  cylindrical
        cart[1] = spher[0] * np.sin(theta)   #y
        cart[2] = spher[2] #z
        """
    else:
        theta = spher[:,1]
        phi = spher[:,2]
        cart[:,0] = spher[:,0] * np.sin(theta) * np.cos(phi) #x spherical
        cart[:,1] = spher[:,0] * np.sin(theta) * np.sin(phi)   #y
        cart[:,2] = spher[:,0] * np.cos(theta) #z
        """
        cart[:,0] = spher[:,0] * np.cos(theta)  #x  cylindrical
        cart[:,1] = spher[:,0] * np.sin(theta)   #y
        cart[:,2] = spher[:,2] #z
        """
    return cart

def find_closest_point_on_line_knee(point1, point2, target_point):
    x_mid = (point1[0] + point2[0]) / 2
    y_mid = (point1[1] + point2[1]) / 2
    z_mid = (point1[2] + point2[2]) / 2
    return np.array([x_mid, y_mid, z_mid])

def no_occlusion_threeDposes(normalised_poses, poses_list, radar_data, left_lifter,right_lifter): #not_occluded poses
    left_inp_pose, right_inp_pose = split_data_left_right(normalised_poses)
    pred_left, _ = left_lifter(left_inp_pose)
    pred_right, _ = right_lifter(right_inp_pose)

    pred_left_right = combine_left_right_pred_1d(pred_left, pred_right, choice='right').reshape(-1, 15)
    pred_left_right[:, 0] = 0.0
    pred_left_right = - pred_left_right - 10 

    pred_3d_pose = torch.cat(((normalised_poses.reshape(-1, 2, 15) * pred_left_right.reshape(-1, 1, 15).repeat(1, 2, 1)).reshape(-1, 30), pred_left_right), dim=1)
    full_3d_pose = pred_3d_pose.reshape(-1, 3, 15) # I reshape it to (1,15,3) with these two commands
    for j in range(pred_left_right.shape[0]):
        full_3d_pose[j,2,:] += 10

        angle = np.radians(radar_data[:,3][j])
        x_values = full_3d_pose[j,0,:] * torch.tensor(np.cos(angle)) + full_3d_pose[j,2,:] * torch.tensor(np.sin(angle))
        full_3d_pose[j,2,:] = - full_3d_pose[j,0,:] * torch.tensor(np.sin(angle)) + full_3d_pose[j,2,:] * torch.tensor(np.cos(angle))
        full_3d_pose[j,0,:] = x_values

        full_3d_pose[j,0,:] -= torch.tensor(radar_data[:,0][j]) #x
        full_3d_pose[j,2,:] = full_3d_pose[j,2,:] - torch.tensor(radar_data[:,1][j]) #z
        full_3d_pose[j,1,:] -= torch.min(full_3d_pose[j,1,[3,6]]) #y
    for i in range(normalised_poses.shape[0]):
        pose = full_3d_pose[i,:,:].T.detach().cpu().numpy()
        
        ankles_limit = 0.5
        knees_limit = 0.2
        distance_Lhip = np.sum(np.square(pose[4,:]))
        distance_Rhip = np.sum(np.square(pose[1,:]))
        distance_Lankle = np.sum(np.square(pose[6,:]))
        distance_Rankle = np.sum(np.square(pose[3,:]))
        closest_point = find_closest_point_on_line(pose[4,:], [pose[4, 0],-10,pose[4,2]], pose[6,:])
        if distance_Lankle > distance_Lhip and distance_Lankle - distance_Lhip < 0.5:
            pose[6,:] = closest_point
        else: 
            if distance_Lankle > distance_Lhip:
                ankle_in_new_sys = pose[6,:] - closest_point 
                spher = appendSpherical_np(ankle_in_new_sys)
                spher[0] = spher[0] - 0.5
                pose[6,:] = to_cartesian(spher) + closest_point

        closest_point = find_closest_point_on_line(pose[1,:], [pose[1,0],-10,pose[1,2]], pose[3,:])
        if distance_Rankle > distance_Rhip and distance_Rankle - distance_Rhip < 0.5:
            pose[3,:] = closest_point
        else:
            if distance_Rankle > distance_Rhip:
                ankle_in_new_sys = pose[3,:] - closest_point 
                spher = appendSpherical_np(ankle_in_new_sys)
                spher[0] = spher[0] - 0.5
                pose[3,:] = to_cartesian(spher) + closest_point

        distance = calculate_shortest_distance(pose[4,:], pose[6,:], pose[5,:])
        closest_point = find_closest_point_on_line_knee(pose[4,:], pose[6,:], pose[5,:])
        if distance < knees_limit: #if LKnee is bent less than average, make it perfectly straight
            pose[5,:] = closest_point
        else:
            knee_in_new_sys = pose[5,:] - closest_point 
            spher = appendSpherical_np(knee_in_new_sys)
            spher[0] = spher[0] - knees_limit   
            pose[5,:] = to_cartesian(spher) + closest_point
            
        distance = calculate_shortest_distance(pose[1,:], pose[3,:], pose[2,:])
        closest_point = find_closest_point_on_line_knee(pose[1,:], pose[3,:], pose[2,:])
        if distance < knees_limit: #if RKnee is bent less than average, make it perfectly straight
            pose[2,:] = closest_point
        else:
            knee_in_new_sys = pose[2,:] - closest_point 
            spher = appendSpherical_np(knee_in_new_sys)
            spher[0] = spher[0] - knees_limit
            pose[2,:] = to_cartesian(spher) + closest_point  

        poses_list.append(pose)
    return poses_list

# If the pose is occluded, the local and global coordinates of it are estimated using this function
def occlusion_threeDposes(normalised_poses, keypoints, poses_list, radar_data, left_lifter,right_lifter,legs_lifter,torso_lifter,left_leg_predictor,right_leg_predictor,left_arm_predictor,right_arm_predictor):
    limbs = [[1,2,3],[4,5,6],[9,10,11],[12,13,14]] #right leg keypoint, leg left keypoints, ...
    left_inp_pose, right_inp_pose = split_data_left_right(normalised_poses)
    inp_legs = normalised_poses.reshape(-1, 2, 15)[:, :, :7].reshape(-1, 14)
    inp_torso = torch.cat((normalised_poses.reshape(-1, 2, 15)[:, :, [0]], normalised_poses.reshape(-1, 2, 15)[:, :, 7:]), dim=2).reshape(-1, 18)

    for i in range(keypoints.shape[0]): #loop over all people
        flag = np.array([0,0,0,0]) #flags to check which limbs are occluded
        for j in range(len(limbs)): #detect which limbs are occluded
            if np.isnan(keypoints[i,limbs[j],:]).any():
                flag[j] = 1
        if np.sum(flag) == 1 and np.isnan(keypoints[i,[0,7,8],:]).any() == False: #only one limb keypoints is occluded and head, pelvis or neck are not occluded
            if flag[0] == 1: #right leg
                pred_left, _ = left_lifter(torch.reshape(left_inp_pose[i,:],(1,18))) 
                pred_torso, _ = torso_lifter(torch.reshape(inp_torso[i,:],(1,18)))
                pred_occluded_z = torch.cat((pred_left[:9],pred_torso[6:]), dim=0) 

                pred_occluded_z[0] = 0.0 
                pred_occluded_z = - pred_occluded_z - 10   

                occluded_poses = normalised_poses[i,:,[0,4,5,6,7,8,9,10,11,12,13,14]] #occluded pose cant contain limb keypoints
                partial_3d_pose = torch.cat(((occluded_poses.reshape(-1, 2, 12) * pred_occluded_z.reshape(-1, 1, 12).repeat(1, 2, 1)).reshape(-1, 24), torch.reshape(pred_occluded_z,(1,12))), dim=1) 
                partial_3d_pose = partial_3d_pose.reshape(-1,36)

                right_leg_predictions = right_leg_predictor(partial_3d_pose) 
                full_3d_pose = combine_pose_and_limb(partial_3d_pose, right_leg_predictions, 'rl')

            if flag[1] == 1: #left leg	
                pred_right, _ = right_lifter(torch.reshape(right_inp_pose[i,:],(1,18))) #predict right side only for one person
                pred_torso, _ = torso_lifter(torch.reshape(inp_torso[i,:],(1,18)))
                pred_occluded_z = torch.cat((pred_right[:6], pred_torso[3:6], pred_right[6:]), dim=0) 

                pred_occluded_z[0] = 0.0 
                pred_occluded_z = - pred_occluded_z - 10           

                occluded_poses = normalised_poses[i,:,[0,1,2,3,7,8,9,10,11,12,13,14]] #occluded pose cant contain limb keypoints
                partial_3d_pose = torch.cat(((occluded_poses.reshape(-1, 2, 12) * pred_occluded_z.reshape(-1, 1, 12).repeat(1, 2, 1)).reshape(-1, 24), torch.reshape(pred_occluded_z,(1,12))), dim=1) 
                partial_3d_pose = partial_3d_pose.reshape(-1,36)

                left_leg_predictions = left_leg_predictor(partial_3d_pose) 
                full_3d_pose = combine_pose_and_limb(partial_3d_pose, left_leg_predictions, 'll')

            if flag[2] == 1: #left arm
                pred_right, _ = right_lifter(torch.reshape(right_inp_pose[i,:],(1,18))) #predict right side only for one person
                pred_legs, _ = legs_lifter(torch.reshape(inp_legs[i,:],(1,14)))
                pred_occluded_z = torch.cat((pred_right[:4], pred_legs[4:7], pred_right[4:]), dim=0) 

                pred_occluded_z[0] = 0.0 
                pred_occluded_z = - pred_occluded_z - 10           

                occluded_poses = normalised_poses[i,:,[0,1,2,3,4,5,6,7,8,12,13,14]] #occluded pose cant contain limb keypoints
                partial_3d_pose = torch.cat(((occluded_poses.reshape(-1, 2, 12) * pred_occluded_z.reshape(-1, 1, 12).repeat(1, 2, 1)).reshape(-1, 24), torch.reshape(pred_occluded_z,(1,12))), dim=1) 
                partial_3d_pose = partial_3d_pose.reshape(-1,36)

                left_arm_predictions = left_arm_predictor(partial_3d_pose) 
                full_3d_pose = combine_pose_and_limb(partial_3d_pose, left_arm_predictions, 'la')


            if flag[3] == 1: #right arm
                pred_left, _ = left_lifter(torch.reshape(left_inp_pose[i,:],(1,18))) #predict right side only for one person
                pred_legs, _ = legs_lifter(torch.reshape(inp_legs[i,:],(1,14)))
                pred_occluded_z = torch.cat((pred_legs[:4], pred_left[1:]), dim=0) 
                #pred_occluded_z = torch.cat((pred_left[0], pred_legs[1:4], pred_left[1:]), dim=0) better but pred_left[0] zero-dimensional tensor cannot be concatenated

                pred_occluded_z[0] = 0.0 
                pred_occluded_z = - pred_occluded_z - 10       

                occluded_poses = normalised_poses[i,:,[0,1,2,3,4,5,6,7,8,9,10,11]] #occluded pose cant contain limb keypoints
                partial_3d_pose = torch.cat(((occluded_poses.reshape(-1, 2, 12) * pred_occluded_z.reshape(-1, 1, 12).repeat(1, 2, 1)).reshape(-1, 24), torch.reshape(pred_occluded_z,(1,12))), dim=1) 
                partial_3d_pose = partial_3d_pose.reshape(-1,36)

                right_arm_predictions = right_arm_predictor(partial_3d_pose) 
                full_3d_pose = combine_pose_and_limb(partial_3d_pose, right_arm_predictions, 'ra')

            full_3d_pose = full_3d_pose.reshape(-1, 3, 15) # I reshape it to (1,15,3) with these two commands
            full_3d_pose[:,2,:] = full_3d_pose[:,2,:] + 10

            angle = np.radians(radar_data[i,3])
            x_values = full_3d_pose[:,0,:] * np.cos(angle) + full_3d_pose[:,2,:] * np.sin(angle)
            full_3d_pose[:,2,:] = - full_3d_pose[:,0,:] * np.sin(angle) + full_3d_pose[:,2,:] * np.cos(angle)
            full_3d_pose[:,0,:] = x_values

            full_3d_pose[:,0,:] -= radar_data[i,0]
            full_3d_pose[:,2,:] = full_3d_pose[:,2,:] + radar_data[i,1]
            full_3d_pose[:,1,:] -= torch.min(full_3d_pose[:,1,[3,6]])
            full_3d_pose = full_3d_pose[0,:,:].T.detach().cpu().numpy()
            
            distance_Lhip = np.sum(np.square(full_3d_pose[4,:]))
            distance_Rhip = np.sum(np.square(full_3d_pose[1,:]))
            distance_Lankle = np.sum(np.square(full_3d_pose[6,:]))
            distance_Rankle = np.sum(np.square(full_3d_pose[3,:]))
            closest_point = find_closest_point_on_line(full_3d_pose[4,:], [full_3d_pose[4, 0],-10,full_3d_pose[4,2]], full_3d_pose[6,:])
            if distance_Lankle > distance_Lhip and distance_Lankle - distance_Lhip < 0.5:
                full_3d_pose[6,:] = closest_point
            else: 
                if distance_Lankle > distance_Lhip:
                    ankle_in_new_sys = full_3d_pose[6,:] - closest_point 
                    spher = appendSpherical_np(ankle_in_new_sys)
                    spher[0] = spher[0] - 0.5
                    full_3d_pose[6,:] = to_cartesian(spher) + closest_point
        
            closest_point = find_closest_point_on_line(full_3d_pose[1,:], [full_3d_pose[1,0],-10,full_3d_pose[1,2]], full_3d_pose[3,:])
            if distance_Rankle > distance_Rhip and distance_Rankle - distance_Rhip < 0.5:
                full_3d_pose[3,:] = closest_point
            else: 
                if distance_Rankle > distance_Rhip:
                    ankle_in_new_sys = full_3d_pose[3,:] - closest_point 
                    spher = appendSpherical_np(ankle_in_new_sys)
                    spher[0] = spher[0] - 0.5
                    full_3d_pose[3,:] = to_cartesian(spher) + closest_point
        
            distance = calculate_shortest_distance(full_3d_pose[4,:], full_3d_pose[6,:], full_3d_pose[5,:])
            closest_point = find_closest_point_on_line(full_3d_pose[4,:], full_3d_pose[6,:], full_3d_pose[5,:])
            if distance < 0.45: #if LKnee is bent less than average, make it perfectly straight
                full_3d_pose[5,:] = closest_point
            else:
                knee_in_new_sys = full_3d_pose[5,:] - closest_point 
                spher = appendSpherical_np(knee_in_new_sys)
                spher[0] = spher[0] - 0.45    #0.45 for knees alone
                full_3d_pose[5,:] = to_cartesian(spher) + closest_point
            
            distance = calculate_shortest_distance(full_3d_pose[1,:], full_3d_pose[3,:], full_3d_pose[2,:])
            closest_point = find_closest_point_on_line(full_3d_pose[1,:], full_3d_pose[3,:], full_3d_pose[2,:])
            if distance < 0.45: #if RKnee is bent less than average, make it perfectly straight
                full_3d_pose[2,:] = closest_point
            else:
                knee_in_new_sys = full_3d_pose[2,:] - closest_point 
                spher = appendSpherical_np(knee_in_new_sys)
                spher[0] = spher[0] - 0.45
                full_3d_pose[2,:] = to_cartesian(spher) + closest_point 

            poses_list.append(full_3d_pose)
    return poses_list

# Function calculates the distance of a person away from the system based on the radar coordinates in global space
def distance_function(coordinates):
    x, y = coordinates[:, 0], coordinates[:, 1]
    angles = np.arctan2(x,y)
    degrees = np.degrees(angles)
    print('degrees',degrees)
    distances = np.sqrt(x**2 + y**2)
    return distances, degrees

# Function to calculate the average height and angle for smooth scaling and rotation of the human pose 
def smooth(data, new_points,angle_dis_storage_1, old_points, old_mem):
    angle_dis_storage = np.vstack([angle_dis_storage_1[0],old_mem[0]])
    angle_storage = np.vstack([angle_dis_storage_1[1],old_mem[1]])
    data = np.vstack([data,old_points])
    kdtree = spatial.KDTree(data)
    indices = []
    for i in range(new_points.shape[0]):
        # Find the closest point in the KD tree
        distance, index = kdtree.query(new_points[i])
        if distance < 0.3: #treshold
            indices.append(index)
        else:
            indices.append(np.nan)
    dist, angle = distance_function(new_points)
    #print('dist, angle',dist, angle)
    non_indices = [num for num in range(data.shape[0]) if num not in indices]
    nan_index = np.ravel(np.argwhere(np.isnan(indices))).tolist()
    if nan_index: #for people that have just appeared and we have no history of data for them!
        #print('IN NAN_INDEX')
        smoothed_reading = np.hstack([new_points[nan_index],dist[nan_index].reshape(-1,1),angle[nan_index].reshape(-1,1)])
        a = np.full([len(nan_index), 6], np.nan)
        new_memory = np.hstack([dist[nan_index].reshape(-1,1),a])
        new_memory_angle = np.hstack([angle[nan_index].reshape(-1,1),a])
        cleaned_indices = [int(x) for x in indices if not np.isnan(x)]
        non_nan_indices = [i for i, val in enumerate(indices) if not np.isnan(val)]
        if cleaned_indices: #for people with previous data
            #print('IN CLEANED_INDICES')
            new_radar_dist = np.nanmean((np.vstack([dist[non_nan_indices],angle_dis_storage[cleaned_indices].T])), axis=0)
            stack = np.deg2rad(np.vstack([angle[non_nan_indices],angle_storage[cleaned_indices].T]))
            new_radar_angle = np.rad2deg(np.arctan2(np.nansum(np.sin(stack),axis=0),np.nansum(np.cos(stack),axis=0)))
            readings_2 = np.hstack([new_points[non_nan_indices],new_radar_dist.reshape(-1,1),new_radar_angle.reshape(-1,1)])
            smoothed_reading = np.vstack([smoothed_reading,readings_2]) #combine
            smoothed_reading = smoothed_reading[[nan_index + non_nan_indices]].reshape(-1,4) #change order
            new_memory_2_dist = np.vstack([dist[non_nan_indices],angle_dis_storage[cleaned_indices].T])[:7].T
            new_memory_2_angle = np.vstack([angle[non_nan_indices],angle_storage[cleaned_indices].T])[:7].T
            new_memory = np.vstack([new_memory, new_memory_2_dist])
            new_memory_angle = np.vstack([new_memory_angle, new_memory_2_angle])
        new_memory_angle = new_memory_angle[nan_index + non_nan_indices]
        new_memory = new_memory[nan_index + non_nan_indices]
        new_memory = np.stack([new_memory,new_memory_angle])
    else:
        #print('IN ELSE')
        new_radar_dist = np.nanmean((np.vstack([dist,angle_dis_storage[indices].T])), axis=0)
        stack = np.deg2rad(np.vstack([angle,angle_storage[indices].T]))
        new_radar_angle = np.rad2deg(np.arctan2(np.nansum(np.sin(stack),axis=0),np.nansum(np.cos(stack),axis=0)))
        new_memory = np.vstack([dist,angle_dis_storage[indices].T])[:7].T
        new_memory_angle = np.vstack([angle,angle_storage[indices].T])[:7].T
        new_memory = np.stack([new_memory,new_memory_angle])
        smoothed_reading = np.hstack([new_points,new_radar_dist.reshape(-1,1),new_radar_angle.reshape(-1,1)])
    return smoothed_reading, new_memory, data[non_indices][:5], np.stack([angle_dis_storage[non_indices],angle_storage[non_indices]])[:, :5, :]

# Function to untilt the upper body keypoints of the human pose from diagional towards camera to upright
def tilt(poses):
    # If the poses variable is a list
    #list = []
    #for i in range(len(poses_list)):
    #    list.append(poses_list[i].detach().numpy())
    #poses = np.array(list)

    # Calculate middle hip keypoints distance and hypotenuse, elevalation angle, tilt angle and correction value, s
    midhip_dis = np.sqrt(poses[:, 0, 0]**2 + poses[:, 0, 2]**2)
    elev_angle = np.arctan((poses[:, 0, 1]) / midhip_dis)
    h = midhip_dis / np.cos(elev_angle)
    tilt_angle = np.arctan((h - midhip_dis) / (poses[:, 0, 1]))
    s = (poses[:, 7, 1] - poses[:, 0, 1]) * np.tan(tilt_angle)

    # Calculate the polar distance and phi angle of each keypoint in each human poses
    polar_dis = np.sqrt(poses[:, :, 0]**2 + poses[:, :, 2]**2)
    phi = np.arctan2(poses[:, :, 2], poses[:, :, 0])

    # Add the correction value, s to polar distance
    for i in range(len(poses)):
        #polar_dis[i, 7:15] = polar_dis[i, 7:15] + s[i]
        polar_dis[i, 7:10] = polar_dis[i, 7:10] + s[i]
        polar_dis[i, 12] = polar_dis[i, 12] + s[i]
        if poses[i, 10, 1] > poses[i, 7, 1]: # left hand elbow rise higher than neck
            polar_dis[i, 10] = polar_dis[i, 10] + s[i]
            polar_dis[i, 11] = polar_dis[i, 11] + s[i]
        if poses[i, 13, 1] > poses[i, 7, 1]: # right hand elbow rise higher than neck
            polar_dis[i, 13] = polar_dis[i, 13] + s[i]
            polar_dis[i, 14] = polar_dis[i, 14] + s[i]

    # Update the human poses coordinate
    poses[:, 7:15, 0] = polar_dis[:, 7:15] * np.cos(phi[:, 7:15])
    poses[:, 7:15, 2] = polar_dis[:, 7:15] * np.sin(phi[:, 7:15])

    return poses

#This function prepares the keypoint for lifting, send them to occlusion_threeDposes or no_occlusion_threeDposes functions, and merges them toghether into one array
def H_threeDposes(keypoints, radar_data, left_lifter,right_lifter,legs_lifter,torso_lifter,left_leg_predictor,right_leg_predictor,left_arm_predictor,right_arm_predictor):
    normalised_poses = normalize_head_test(torch.from_numpy(np.asarray(keypoints[:,:,:2], dtype=np.float32)).permute(0, 2, 1),np.asarray(radar_data[:,2], dtype=np.float32))
    poses_list = []

    no_occlusion_keypoints_indexes = np.all(np.all(~np.isnan(keypoints), axis=1), axis=1)
    no_occlusion_normalised_poses = normalised_poses[no_occlusion_keypoints_indexes]
    occlusion_keypoints_indexes = ~ no_occlusion_keypoints_indexes
    occlusion_normalised_poses = normalised_poses[occlusion_keypoints_indexes]
    keypoints = keypoints[occlusion_keypoints_indexes,:,:]

    if no_occlusion_keypoints_indexes.any() == True:
        poses_list = no_occlusion_threeDposes(no_occlusion_normalised_poses, poses_list, radar_data[no_occlusion_keypoints_indexes], left_lifter,right_lifter)
    if occlusion_keypoints_indexes.any() == True:
        poses_list = occlusion_threeDposes(occlusion_normalised_poses, keypoints, poses_list, radar_data[occlusion_keypoints_indexes], left_lifter,right_lifter,legs_lifter,torso_lifter,left_leg_predictor,right_leg_predictor,left_arm_predictor,right_arm_predictor)

    if len(poses_list) != 0:
        poses = np.array(poses_list)
        #print('poses list', poses, poses.shape)
        #print('')
        poses = tilt(poses)

        return poses[:,[8,7,9,10,11,12,13,14,4,5,6,1,2,3,0],:]

#This function converts radar coordinates into camera coordinates, and calibrates radar coordinates
def radar_data_transform(xzs):
    radar_1_transform = np.array([[1.23373116e+02,  0.0,  1.07756048e+03], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    #radar_2_transform = np.array([[205.58671755,  0.0,  382.87968182], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    #radar_3_transform = np.array([[1.41695160e+02,  0.0,  1.65088222e+03], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    radar_2_transform = np.array([[120.14739353,  0.0,  572.83728603], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    radar_3_transform_1 = np.array([[95.6210308,  0.0,  143.68054037], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    radar_3_transform_2 = np.array([[58.77601971,  0.0,  1828.77367858], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    radar_4_transform = np.array([[279.38034509,  0.0,  1508.26418375], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    intrinsic_par_1 = np.array([[0.94634545,  0.07645529,  0.01693388], [-0.00895847,  1.00596408, -0.02776186], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    intrinsic_par_2 = np.array([[1.06437966, -0.00258283,  0.10063119], [-0.05842811,  0.9611548 ,-0.04858848], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    intrinsic_par_3 = np.array([[0.96655344, -0.03690172,  0.04259001], [0.02669844,  0.93537586,0.10827645], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    intrinsic_par_4 = np.array([[1.0, 0.0,  0.0], [0.0,  1.0, 0.0], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    if xzs[0] is None and xzs[1] is None and xzs[2] is None and xzs[3] is None:
        print("NO PEOPLE DETECTED BY THE RADARS")
        return None, None
    else:
        if xzs[0] is not None: #radar 1 if there is at least one person detected
            coords_for_better = np.ones(shape=(xzs[0].shape[0], 3)) 
            for i in range(xzs[0].shape[0]): #paste radar reading to correct format
                coords_for_better[i] = [xzs[0][i][0],xzs[0][i][1],1]
            transformed_1 = (radar_1_transform @ coords_for_better.T).T #improve radar reading
            coords_better = (intrinsic_par_1 @ coords_for_better.T).T #transform radar reading to the x-coordinate of camera for matching
            coords_1 = coords_better[:,:2]
        else:
            coords_1 = None
            transformed_1 = [np.nan, np.nan, np.nan]
        if xzs[1] is not None:
            coords_for_better = np.ones(shape=(xzs[1].shape[0], 3))
            for i in range(xzs[1].shape[0]):
                coords_for_better[i] = [xzs[1][i][0],xzs[1][i][1],1]
            transformed_2 = (radar_2_transform @ coords_for_better.T).T
            coords_better = (intrinsic_par_2 @ coords_for_better.T).T
            coords_2 = coords_better[:,:2]
        else:
            coords_2 = None
            transformed_2 = [np.nan, np.nan, np.nan]
        if xzs[2] is not None:
            coords_for_better = np.ones(shape=(xzs[2].shape[0], 3))
            for i in range(xzs[2].shape[0]):
                coords_for_better[i] = [xzs[2][i][0],xzs[2][i][1],1]
            transformed_3_1 = (radar_3_transform_1 @ coords_for_better.T).T
            transformed_3_2 = (radar_3_transform_2 @ coords_for_better.T).T
            coords_better = (intrinsic_par_3 @ coords_for_better.T).T
            coords_3 = coords_better[:,:2]
        else:
            coords_3 = None
            transformed_3_1 = [np.nan, np.nan, np.nan]
            transformed_3_2 = [np.nan, np.nan, np.nan]
        if xzs[3] is not None:
            coords_for_better = np.ones(shape=(xzs[3].shape[0], 3))
            for i in range(xzs[3].shape[0]):
                coords_for_better[i] = [xzs[3][i][0],xzs[3][i][1],1]
            transformed_4 = (radar_4_transform @ coords_for_better.T).T
            coords_better = (intrinsic_par_4 @ coords_for_better.T).T
            coords_4 = coords_better[:,:2]
        else:
            coords_4 = None
            transformed_4 = [np.nan, np.nan, np.nan]

        transformed = np.vstack([transformed_1, transformed_2, transformed_3_1, transformed_3_2, transformed_4])
        transformed = transformed[~np.isnan(transformed).any(axis=1)]
        #print('Radar 1: ',coords_1,'. Radar 2: ',coords_2,'. Radar 3: ',coords_3)
        coords = under_same_reference(coords_1, coords_2, coords_3, coords_4)
        return coords, transformed

def mirrorImage( a, b, c, x1, y1):
    temp = -2 * (np.multiply(a, x1) + np.multiply(b, y1) + c) / (a * a + b * b)
    x = temp * a + x1
    y = temp * b + y1 
    return (x, y)

def xyz_to_lonlat(xyz):
    """
    Normalise XYZ coordinates an convert them to longitude and latitude.
    """
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x, y, z = xyz_norm[..., 0:1], xyz_norm[..., 1:2], xyz_norm[..., 2:]
 
    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return np.concatenate([lon, lat], axis=-1)
 
def lonlat_to_pixel(lonlat, width, height):
    """
    Convert longitude and latitude coordinates to pixel coordinates.
    """
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (width - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (height - 1)
    return np.concatenate([X, Y], axis=-1)
 
def transform_keypoints(keypoints, fov, theta, phi, width, height, cx, cy):
    """
    Transform keypoints based on camera parameters and rotation angles.

    Parameters:
    keypoints: Keypoints to transform
    fov: Field of view of the camera (in degrees)
    theta, phi: Rotation angles 
    width, height: Image dimensions 
    cx, cy: Camera center coordinates 

    Returns:
    Transformed keypoints in pixel coordinates.
    """
    # Compute intrinsic camera matrix
    f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)
    K_inv = np.linalg.inv(K)
 
    keypoints_homogeneous = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
    keypoints_transformed = np.dot(keypoints_homogeneous, K_inv.T)
 
    # Rotate the keypoints around the y-axis and then around the x-axis given theta and phi angles
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = R2 @ R1
 
    keypoints_transformed = np.dot(keypoints_transformed, R.T)
 
    # Convert to lon/lat and then to pixel coordinates
    lonlat = xyz_to_lonlat(keypoints_transformed)
    XY = lonlat_to_pixel(lonlat, width, height).astype(np.float32)
 
    return XY

def scale_keypoints(keypoints, scale_factor):
    return keypoints*scale_factor

def undistort_keypoints(array_keypoints):
    phi = 0
    height, width = 960, 1920
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    denominator = (width^2+height^2)**0.5
 
    undistorted_array = np.empty_like(array_keypoints)
 
    for i in range(len(array_keypoints)):
        keypoints = array_keypoints[i, :, :]

         # Skip figures that their torso point is NaN
        if np.isnan(keypoints[0]).all():
            undistorted_array[i, :, :] = keypoints
            continue
 
        keypoints_copy = keypoints.copy()

        # Copy and filter out NaN keypoints
        keypoints = keypoints[~np.isnan(keypoints).any(axis=1)]
        non_nan_indices = np.arange(len(keypoints_copy))[~np.isnan(keypoints_copy).any(axis=1)]
        
        # Calculate mirrored keypoints and set centre point to be the same
        center_point = keypoints[0]
        mirrored = 2 * center_point - keypoints
        mirrored[0] = center_point
        keypoints = mirrored
 
        x_coordinates, y_coordinates = keypoints[:, 0], keypoints[:, 1]

        mid_theta = np.mean(x_coordinates)
        theta = (mid_theta/width*360) - 180 #Calculate theta in degrees
 
        # Compute the minimum and maximum values for x and y coordinates
        min_x, max_x = np.min(x_coordinates), np.max(x_coordinates)
        min_y, max_y = np.min(y_coordinates), np.max(y_coordinates)
 
        # Calculate the width and height based on the keypoints
        width_person = max_x - min_x
        height_person = max_y - min_y
 
        person = max(width_person, height_person)

        # Calculate and adjust field of view based on theta and the person's largest dimension
        K = 0.12 * np.cos(np.radians(theta)) + 1

        fov = K*(1.4*np.arctan(person/denominator)*(180/np.pi))
 
        # Perform the transformation
        transformed_keypoints = transform_keypoints(keypoints, fov, theta, phi, width, height, cx, cy)
 
        x_transformed = transformed_keypoints[:, 0]
        y_transformed = transformed_keypoints[:, 1]
 
        min_x_tr, max_x_tr = np.min(x_transformed), np.max(x_transformed)
        min_y_tr, max_y_tr = np.min(y_transformed), np.max(y_transformed)
 
        w_person_tr = max_x_tr-min_x_tr
        h_person_tr = max_y_tr-min_y_tr

        if w_person_tr > 1900 or h_person_tr > 950:
            undistorted_array[i, :, :] = keypoints_copy
            continue
 
        person_transformed = max(w_person_tr, h_person_tr)
 
        transformed_keypoints = scale_keypoints(transformed_keypoints, (person/person_transformed))
 
        # Adjusting the final position of the keypoints
        center_p = transformed_keypoints[0]
        transformed_keypoints = 2 * center_p - transformed_keypoints
        transformed_keypoints[0] = center_p
        remove_distance = center_p - center_point
        transformed_keypoints = transformed_keypoints - remove_distance
 
        # Insert NaN keypoints back into the transformed array, in their original positions
        result = np.zeros_like(keypoints_copy)
        result[non_nan_indices] = transformed_keypoints
        result[result[:, :] == [0., 0.]] = np.nan
 
        undistorted_array[i, :, :] = result

    return undistorted_array

######################################## MAIN THREAD ###################################################

if __name__ == '__main__':
    print("MULTI-PERSON 3D HUMAN POSE ESTIMATION ALGORITHM STARTING ...")
    ############################### START GUI PROCESS #######################################
    subprocess.Popen("C:/Apps/Anaconda/envs/visualiser/python.exe gui.py")
    time.sleep(2)
    address = ('localhost', 6000)
    conn = Client(address, authkey=None)
    ############################### START RADARS SESSIONS ####################################
    radar_event = Event()
    radar_queue = Queue()
    radar_process1 = Process(target=radar_session, args=(1, command_port1, data_port1, radar_event, radar_queue,))
    radar_process1.start()
    radar_process2 = Process(target=radar_session, args=(2, command_port2, data_port2, radar_event, radar_queue,))
    radar_process2.start()
    radar_process3 = Process(target=radar_session, args=(3, command_port3, data_port3, radar_event, radar_queue,))
    radar_process3.start()
    radar_process4 = Process(target=radar_session, args=(4, command_port4, data_port4, radar_event, radar_queue,))
    radar_process4.start()
    ##################### NEURAL NETWORK MODEL INSTANTIATION ################################

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load the weights of the models
    left_lifter = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=9, use_dropout=False, d_rate=0.25,num_heads=2)#.cuda()   Could not get cuda to work
    right_lifter = Attention_Left_Right_Lifter(use_batchnorm=False, num_joints=9, use_dropout=False, d_rate=0.25,num_heads=2)#.cuda()  doesn't matter for me
    legs_lifter = Attention_Leg_Lifter(use_batchnorm=False, num_joints=7, use_dropout=False, d_rate=0.25)#.cuda()
    torso_lifter = Attention_Torso_Lifter(use_batchnorm=False, num_joints=9, use_dropout=False, d_rate=0.25)#.cuda()

    left_lifter.load_state_dict(torch.load('weights/left_side_lifter_openpose.pt'))
    right_lifter.load_state_dict(torch.load('weights/right_side_lifter_openpose.pt'))
    legs_lifter.load_state_dict(torch.load('weights/legs_lifter_openpose.pt'))
    torso_lifter.load_state_dict(torch.load('weights/torso_lifter_openpose.pt'))

    left_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=12)
    right_leg_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=12)
    left_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=12)
    right_arm_predictor = Occluded_Limb_Predictor(use_batchnorm=False, num_joints=12)

    left_leg_predictor.load_state_dict(torch.load('occlusion_model_weights/left_leg_estimator.pt'))
    right_leg_predictor .load_state_dict(torch.load('occlusion_model_weights/right_leg_estimator.pt'))
    left_arm_predictor.load_state_dict(torch.load('occlusion_model_weights/left_arm_estimator.pt'))
    right_arm_predictor.load_state_dict(torch.load('occlusion_model_weights/right_arm_estimator.pt'))
    ################################### MAIN CODE #################################################
    cap = None
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH'] = os.environ[
                                         'PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", default="../../../examples/media/",
                            help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        parser.add_argument("--camera_height", type=float, default=1.06, help="Set camera height")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        cap = cv2.VideoCapture(0)
        # Camera calibration coefficients is load but unused
        mtx, dist = load_coefficients('calibration_chessboard.yml')

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        k = 0
        print("MULTI-PERSON 3D HUMAN POSE ESTIMATION ALGORITHM RUNNING ...")
        while True:
            start = time.time()
            # Capture video frame from camera
            ret, frame = cap.read()
            img = frame
            h = 0
            w = 0
            # Expand camera frame view by duplicating left part of the frame to right side of the frame
            cropped_img = img[(h):((h+960)), (w):((w+45))]
            frame_new = np.hstack((frame, cropped_img))
            radar_event.set()
            xzs = []
            for i in range(4):
                l, d = radar_queue.get()
                xzs.append((l,d))
            xzs.sort(key=lambda x: x[0])
            xzs = list(map(lambda x: x[1], xzs))
            coords, transformed = radar_data_transform(xzs) # Transform radar data for matching and better radar readings
            datum = op.Datum()
            datum.cvInputData = frame_new
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            if datum.poseKeypoints is not None and coords is not None: # If OpenPose detects a person
                pixel_keypoints = datum.poseKeypoints[:, [8, 9, 10, 11, 12, 13, 14, 1, 0, 5, 6, 7, 2, 3, 4], :2] #choose only 15 x,y keypoints

                pixel_keypoints[pixel_keypoints[:, :] == [0., 0.]] = np.nan

                undistorted_pixel_keypoints = undistort_keypoints(pixel_keypoints)
                pixel_keypoints = undistorted_pixel_keypoints

                indexes, index = matching(pixel_keypoints, transformed) # Match camera- and radar-detected individuals
                
                pixel_keypoints = pixel_keypoints[indexes]
                
                #m = (pixel_keypoints[:,7,1] - pixel_keypoints[:,0,1]) / (pixel_keypoints[:,7,0] - pixel_keypoints[:,0,0])
                #b = pixel_keypoints[:,0,1] - m * pixel_keypoints[:,0,0]
                #m = m.reshape(-1,1)
                #b = b.reshape(-1,1)

                #A = mirrorImage(m,-1,b,pixel_keypoints[:,:,0],pixel_keypoints[:,:,1])
                #pixel_keypoints = np.stack((A[0], A[1]), axis=2)

                if len(index) != 0: # If more than one individual was matched then calculate its global 3D coordinates and display them
                    #calculate the average distance and angle reading (based on previous readings)
                    if k == 0: #first frame
                        coord_storage = coords[index]
                        dis_angle = np.full([2,coords[index].shape[0], 7], np.nan) 
                        radar_coords_f, dis_angle, radar_coords_no_pair, dis_angle_no_pair = smooth(coord_storage,coord_storage,dis_angle,coord_storage[0],np.full([2, 1, 7], np.nan))
                        k = k + 1
                    else:
                        radar_coords_f, dis_angle, radar_coords_no_pair, dis_angle_no_pair = smooth(coord_storage,coords[index],dis_angle, radar_coords_no_pair, dis_angle_no_pair)
                        coord_storage = coords[index]

                    #calculate the 3D pose coordinates, and correct global localisation of each detected individual
                    output_poses = H_threeDposes(pixel_keypoints, radar_coords_f, left_lifter,right_lifter,legs_lifter,torso_lifter,left_leg_predictor,right_leg_predictor,left_arm_predictor,right_arm_predictor)
   
                    #visualise poses                 
                    conn.send(output_poses)
                    cv2.imshow('output', datum.cvOutputData)
                    end = time.time()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    conn.send('close')
                    conn.close()
                    break

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        cap.release()
        conn.send('close')
        conn.close()
        sys.exit(-1)

