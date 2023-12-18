#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from copy import *
import math
import open3d as o3d
import torch
from scene import Scene
import numpy as np
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
from scipy.spatial.transform import Rotation
from copy import deepcopy
from typing import NamedTuple
from scene.cameras import Camera
imageio.plugins.freeimage.download()
from scipy.optimize import minimize
import torch as th
from typing import Callable, List, Optional, Set
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import json

######################szh add for render######################
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
def l2_norm(a: th.Tensor, b: th.Tensor) -> float:
    return ((a - b) * (a - b)).sum().item()
# same as ngp
def spline(t, p0, p1, p2, p3):
    tt = t * t
    ttt = tt * t
    a = (1-t)*(1-t)*(1-t)*(1./6.)
    b = (3*ttt - 6*tt + 4)*(1./6.)
    c = (-3*ttt + 3*tt + 3*t + 1)*(1./6.)
    d = ttt*(1./6.)
    return a*p0 + b*p1 + c*p2 + d*p3
# Compute anticlockwise arch distance
# Input: two phi angles
# Output: anticlockwise arch distance
def compute_arch_distance(phi1, phi2):
    phi1 = phi1 % (2 * np.pi)
    phi2 = phi2 % (2 * np.pi)
    phi2 += 2 * np.pi
    phi_dis = phi2 - phi1
    phi_dis = phi_dis % (2 * np.pi)
    return phi_dis
# same as ngp
def spline_quat(t, p0, p1, p2, p3):
    tt = t * t
    ttt = tt * t
    a = (1-t)*(1-t)*(1-t)*(1./6.)
    b = (3*ttt - 6*tt + 4)*(1./6.)
    c = (-3*ttt + 3*tt + 3*t + 1)*(1./6.)
    d = ttt*(1./6.)
    x = a*p0
    if np.dot(p0, p1) < 0:
        p1 = -p1
    x = x + b*p1
    if np.dot(p1, p2) < 0:
        p2 = -p2
    x = x + c*p2
    if np.dot(p2, p3) < 0:
        p3 = -p3
    x = x + d*p3
    return x

def farthest_distance_sampling(
    features: th.Tensor,
    k: int = 10,
    initial_idx: Optional[int] = None,
    dist_func: Optional[Callable[[th.Tensor, th.Tensor], float]] = None,
) -> List[int]:
    if features.shape[0] <= k:
        return list(range(features.shape[0]))

    if initial_idx is None:
        initial_idx = 0

    if dist_func is None:
        dist_func = l2_norm

    selected_idx: Set[int] = {initial_idx}
    for _ in range(k - 1):
        max_dist: float = -1.0
        current_idx: int = -1

        # from remaining points, select one with maximum distance to the selected set of points
        for j in range(features.shape[0]):
            if j in selected_idx:
                continue
            # distance from a point A to a set of points S in defined as
            # the minimum distance of point A to all the points in S
            dist: List[float] = []
            for id in selected_idx:
                d = dist_func(features[j], features[id])
                dist.append(d)
            dist_to_set = np.min(dist)
            if dist_to_set > max_dist:
                max_dist = dist_to_set
                current_idx = j
        if current_idx >= 0:
            selected_idx.add(current_idx)
    selected_indices: List[int] = list(selected_idx)
    selected_indices.sort()
    return selected_indices
def set_look_at(camera_position, target_position, up_vector):
    forward = target_position - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(up_vector, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = -forward
    view_matrix[:3, 3] = camera_position
    return view_matrix
def distance(x, os, ds):
    '''Returns the sum distance between point x and all rays'''
    total_distance = 0
    for o, d in zip(os, ds):
        a = o
        b = o + d
        ab = b - a
        ap = x - a
        total_distance += np.linalg.norm(ap - np.dot(ap, ab/np.dot(ab, ab))*ab)
    return total_distance

def find_min_distance_point(os, ds):
    '''Finds a point in 3D space that has the minimum sum distance to all rays'''
    initial_guess = np.random.rand(3)
    result = minimize(distance, initial_guess, args=(os, ds))
    return result.x
def extract_lookat_up_eye(camera_matrix):
    # Extract camera direction, right direction, up direction, and camera position
    direction = -camera_matrix[:3, 2]
    right = camera_matrix[:3, 0]
    up = camera_matrix[:3, 1]
    position = camera_matrix[:3, 3]
    # Calculate lookat, up, and eye
    lookat = position + direction
    eye = position
    up = up / np.linalg.norm(up)  # Normalize the up vector
    return lookat, up, eye
def compute_object_center(cam_matrixes):
    # Compute up, front, right
    # Compute up by averaging up vectors
    up = np.array([0.0, 0.0, 0.0])
    for cam_matrix in cam_matrixes:
        up += cam_matrix[:3,1]
    up /= len(cam_matrixes)
    up /= np.linalg.norm(up)

    # Compute front by averaging front vectors
    front = np.array([0.0, 0.0, 0.0])
    for cam_matrix in cam_matrixes:
        front += cam_matrix[:3,2]
    front /= len(cam_matrixes)
    front /= np.linalg.norm(front)

    right = np.cross(front, up)
    right /= np.linalg.norm(right)
    front = np.cross(up, right)

    # Compute object center by the intersection of camera rays
    # Find the point that the sum distance to all camera front vectors is minimal by solving a linear system
    os = []
    ds = []
    for cam_matrix in cam_matrixes:
        o = cam_matrix[:3,3]
        d = cam_matrix[:3,2]
        os.append(o)
        ds.append(d)
    center = find_min_distance_point(os, ds)

    # Compute object center by averaging camera centers
    # center = np.array([0.0, 0.0, 0.0])
    # for cam_matrix in cam_matrixes:
    #     center += cam_matrix[:3,3]
    # center /= len(cam_matrixes)

    # Compute mean radius
    radius = 0.0
    for cam_matrix in cam_matrixes:
        radius += np.linalg.norm(cam_matrix[:3,3] - center)
    radius /= len(cam_matrixes)

    return up, front, right, center, radius
# Render the panorama
# Input: camera matrixes
# Output: a circle/arch of camera matrixes
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
def compute_panorama_path(cam_matrixes,num):
    tmp_cam =cam_matrixes.copy()
    cam_matrixes = gaussiancamera_to_ngptransform(cam_matrixes)
    up, front, right, center, radius = compute_object_center(cam_matrixes)

    # Compute average theta by camera positions and up vector
    theta = 0
    for cam_matrix in cam_matrixes:
        pos = cam_matrix[:3,3] - center
        cosTheta = np.dot(pos, up) / np.linalg.norm(pos)
        theta += np.arccos(cosTheta)
    theta /= len(cam_matrixes)

    # Compute phi by camera positions and up vector
    phi = []
    for cam_matrix in cam_matrixes:
        pos = cam_matrix[:3,3] - center
        paral = pos - np.dot(pos, up) / np.linalg.norm(pos) * up
        x = np.array(np.dot(paral, right))
        y = np.array(np.dot(paral, front))
        phi.append(np.arctan2(y, x))

    # Sort phi and get cooresponding index
    sorted_phi = sorted(enumerate(phi), key=lambda x: x[1])
    delta_phi = [sorted_phi[i+1][1] - sorted_phi[i][1] for i in range(len(sorted_phi) - 1)]
    delta_phi.append(sorted_phi[0][1] + 2 * np.pi - sorted_phi[-1][1])
    # find the largest gap
    max_gap = 0
    max_gap_index = 0
    for i in range(len(delta_phi)):
        if delta_phi[i] > max_gap:
            max_gap = delta_phi[i]
            max_gap_index = i
    # start from the largest gap(skip the gap)
    sorted_phi = sorted_phi[max_gap_index + 1:] + sorted_phi[:max_gap_index + 1]
    index = [i[0] for i in sorted_phi]

    phi1 = sorted_phi[0][1]
    phi2 = sorted_phi[-1][1]
    phi3 = sorted_phi[1][1]
    phi_start = phi1
    phi_end = phi2
    if compute_arch_distance(phi1, phi2) < compute_arch_distance(phi1, phi3):
        phi_start = phi2
        phi_end = phi1
    phi_end = phi_start // (2 * np.pi) * 2 * np.pi + phi_end % (2 * np.pi)
    if phi_end < phi_start:
        phi_end += 2 * np.pi
    if phi_end - phi_start > 1.5 * np.pi:
        phi_start = 0
        phi_end = 2 * np.pi

    # Generate camera matrixes of a circle, with latitude theta and up vector
    cam_matrixes = []
    for phi in np.linspace(phi_start, phi_end, num):
        pos = center + radius * np.sin(theta) * np.cos(phi) * right + radius * np.sin(theta) * np.sin(phi) * front + radius * np.cos(theta) * up
        cam_matrix = np.eye(4)
        # output_json["trajectList"].append({"theta":theta,"phi":phi})
        cam_matrix[:3, 3] = pos
        cam_matrix[:3, 2] = center - pos
        cam_matrix[:3, 2] /= np.linalg.norm(cam_matrix[:3, 2])
        cam_matrix[:3, 0] = np.cross(-up, cam_matrix[:3, 2])
        cam_matrix[:3, 0] /= np.linalg.norm(cam_matrix[:3, 0])
        cam_matrix[:3, 1] = np.cross(cam_matrix[:3, 2], cam_matrix[:3, 0])
        cam_matrix[:3, 1] /= np.linalg.norm(cam_matrix[:3, 1])
        cam_matrix[:3, 1] = -cam_matrix[:3, 1]
        cam_matrix[:3, 2] = -cam_matrix[:3, 2]
        cam_matrixes.append(cam_matrix)
    lookat, up, eye = extract_lookat_up_eye(np.linalg.inv(cam_matrixes[0]))
    # output_json["init_camera"] = {"up":up.tolist(),"target":lookat.tolist(),"camera":eye.tolist()}
    output_json = {}
    output_json_ios = {}
    output_json["frames"] = []
    output_json_ios["frames"] = []
    output_json["fx"] = fov2focal(tmp_cam[0].FoVx,tmp_cam[0].original_image.shape[2])
    output_json["fy"] = fov2focal(tmp_cam[0].FoVy,tmp_cam[0].original_image.shape[1])
    output_json_ios["fx"] = fov2focal(tmp_cam[0].FoVx,tmp_cam[0].original_image.shape[2])
    output_json_ios["fy"] = fov2focal(tmp_cam[0].FoVy,tmp_cam[0].original_image.shape[1])
    for i in range(len(cam_matrixes)):
        temp_camera = cam_matrixes[i]
        flip_mat = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Multiply the camera matrix by the flip matrix
        cam_matrix = np.matmul(temp_camera, flip_mat)
        cam_matrix[0, :3] *= -1
        cam_matrix[1, :3] *= -1
        cam_matrix[2, :3] *= -1
        cam_matrix = np.linalg.inv(cam_matrix)
        cam_matrix = cam_matrix.T
        output_json["frames"].append({"idx":"{}".format(i),"transform_matrix":cam_matrix.tolist()})
    # for i in range(len(cam_matrixes)):
    #     temp_camera = cam_matrixes[i]
    #     flip_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    #     fl
    cam_matrixes = my_ngptransform_to_gaussiancamera(cam_matrixes)
    for i in range(len(cam_matrixes)):
        Rotation = cam_matrixes[i].R
        Translation = cam_matrixes[i].T
        m_T = -1 *Translation.reshape(3)
        m_R = np.transpose(Rotation)
        cam = np.eye(4)
        cam[:3,:3] = m_R
        cam[:3,3] = m_T
        cam = np.linalg.inv(cam)
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        cam = np.matmul(cam, flip_mat)
        cam[0] *=-1
        cam[2] *=-1
        output_json_ios["frames"].append({"idx":"{}".format(i),"transform_matrix":cam.tolist()})
    return cam_matrixes,output_json,output_json_ios,center




def interpolate_paronma(cameras):
    up, front, right, center, radius = compute_object_center(cameras)
    # z_height = cameras[0].T[2] # We'll keep all cameras at this z-coordinate

    # Create a sorted list of (index, camera), sorted by their angle in the xy-plane around the center
    sorted_cameras = sorted(
        ((idx, camera) for idx, camera in enumerate(cameras)), 
        key=lambda idx_camera: np.arctan2(idx_camera[1].T[1] - center[1], idx_camera[1].T[0] - center[0])
    )
    z_height = sorted_cameras[0][1].T[2] # We'll keep all cameras at this z-coordinate
    # Create the new interpolated cameras
    interpolated_cameras = []
    for i in range(len(sorted_cameras)):
        # Interpolate position
        t1 = sorted_cameras[i-1][1].T
        t2 = sorted_cameras[i][1].T
        t3 = sorted_cameras[(i+1)%len(sorted_cameras)][1].T
        t4 = sorted_cameras[(i+2)%len(sorted_cameras)][1].T
        num_per_frame = max(300 // len(sorted_cameras), 1)
        for j in range(num_per_frame):  # Generate 60 interpolated cameras between each pair of original cameras
            s = j / num_per_frame  # Normalized interpolation factor
            t = (t2 + s * (t3 - t2))  # Linear interpolation
            t[2] = z_height  # Force z-coordinate to match the first camera

            # Interpolate rotation
            r1 = R.from_matrix(sorted_cameras[i-1][1].R).as_quat()
            r2 = R.from_matrix(sorted_cameras[i][1].R).as_quat()
            r3 = R.from_matrix(sorted_cameras[(i+1)%len(sorted_cameras)][1].R).as_quat()
            r4 = R.from_matrix(sorted_cameras[(i+2)%len(sorted_cameras)][1].R).as_quat()
            q = R.from_quat((r2 + s * (r3 - r2))).as_matrix()  # Linear interpolation
            interpolation_info = Interpolation_Info(R = q, T = t)
            interpolated_cameras.append(interpolation_info)

    return interpolated_cameras
def gaussiancamera_to_ngptransform(cam_matrixes):
    json_cameras = []
    for i in range(0,len(cam_matrixes)):
        m_R = np.transpose(cam_matrixes[i].R)
        m_T = cam_matrixes[i].T.reshape(3,1) 
        combined_matrix = np.hstack((m_R, m_T))
        row_one = np.array([0, 0, 0, 1]).reshape(1, 4)
        m_M = np.vstack((combined_matrix, row_one))
        m_M = np.linalg.inv(m_M)
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        # Multiply the camera matrix by the flip matrix
        cam_matrix = np.matmul(m_M, flip_mat)
        # cam_matrix[0], cam_matrix[2] = -cam_matrix[2],cam_matrix[0]
        # cam_matrix = cam_matrix[[2, 0, 1, 3], :]
        json_cameras.append(cam_matrix)
    return json_cameras
def my_ngptransform_to_gaussiancamera(cam_matrixes):
    gaussian_cameras = []
    # Loop through the input camera matrices
    for i in range(0, len(cam_matrixes)):
        cam_matrix = cam_matrixes[i]
        ################### Convert NGP transforms to COLMAP transforms. ###################
        # Create a flip matrix to reverse the y and z axes
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        # Multiply the camera matrix by the flip matrix
        cam_matrix = np.matmul(cam_matrix, flip_mat)
        # Calculate the inverse of the camera matrix
        cam_matrix = np.linalg.inv(cam_matrix)
        
        ################### Realize the Gaussian's readColmap function to convert COLMAP transforms into Gaussian transforms. ###################
        # Extract the rotation and translation components from the camera matrix
        cam_matrix_R = cam_matrix[:3, :3]  # Rotation matrix
        cam_matrix_T = cam_matrix[:3, 3]    # Translation vector
        m_R = np.transpose(cam_matrix_R)
        m_T = cam_matrix_T
        # Create an Interpolation_Info object with rotation and translation components
        interpolation_info = Interpolation_Info(R = m_R, T = m_T)

        
        # Append the interpolation_info to the list of gaussian_cameras
        gaussian_cameras.append(interpolation_info)
    
    # Return the list of gaussian_cameras
    return gaussian_cameras    
def ngptransform_to_gaussiancamera(cam_matrixes):
    gaussian_cameras = []
    
    # Loop through the input camera matrices
    for i in range(0, len(cam_matrixes)):
        cam_matrix = cam_matrixes[i]
        ################### Convert from left-handed coordinate system to right-handed coordinate system. ###################
        cam_matrix = cam_matrix[[1, 2, 0, 3], :]
        cam_matrix[0], cam_matrix[2] = cam_matrix[2],-cam_matrix[0]
        
        ################### Convert NGP transforms to COLMAP transforms. ###################
        # Create a flip matrix to reverse the y and z axes
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        # Multiply the camera matrix by the flip matrix
        cam_matrix = np.matmul(cam_matrix, flip_mat)
        # Calculate the inverse of the camera matrix
        cam_matrix = np.linalg.inv(cam_matrix)
        
        ################### Realize the Gaussian's readColmap function to convert COLMAP transforms into Gaussian transforms. ###################
        # Extract the rotation and translation components from the camera matrix
        cam_matrix_R = cam_matrix[:3, :3]  # Rotation matrix
        cam_matrix_T = cam_matrix[:3, 3]    # Translation vector
        m_R = np.transpose(cam_matrix_R)
        m_T = cam_matrix_T
        # Create an Interpolation_Info object with rotation and translation components
        interpolation_info = Interpolation_Info(R = m_R, T = m_T)

        
        # Append the interpolation_info to the list of gaussian_cameras
        gaussian_cameras.append(interpolation_info)
    
    # Return the list of gaussian_cameras
    return gaussian_cameras
def interpolate_views(cam_matrixes,radius_scale=1.0):
    up, front, right, center, radius = compute_object_center(cam_matrixes)
    positions = []
    for i in range(len(cam_matrixes)):
        positions.append(cam_matrixes[i].T)
    sampled_indices = farthest_distance_sampling(np.array(positions), min(10, len(positions)))
    sampled_cam_matrixes = []
    for index in sampled_indices:
        sampled_cam_matrixes.append(cam_matrixes[index])
    phi = []
    for cam_matrix in sampled_cam_matrixes:
        pos = cam_matrix.T - center
        paral = pos - np.dot(pos, up) / np.linalg.norm(pos) * up
        x = np.array(np.dot(paral, right))
        y = np.array(np.dot(paral, front))
        phi.append(np.arctan2(y, x))
    sorted_phi = sorted(enumerate(phi), key=lambda x: x[1])
    delta_phi = [sorted_phi[i+1][1] - sorted_phi[i][1] for i in range(len(sorted_phi) - 1)]
    delta_phi.append(sorted_phi[0][1] + 2 * np.pi - sorted_phi[-1][1])
    max_gap = 0
    max_gap_index = 0
    for i in range(len(delta_phi)):
        if delta_phi[i] > max_gap:
            max_gap = delta_phi[i]
            max_gap_index = i
    sorted_phi = sorted_phi[max_gap_index + 1:] + sorted_phi[:max_gap_index + 1]
    index = [i[0] for i in sorted_phi]
    sorted_cam_matrix = []
    for idx in index:
        sorted_cam_matrix.append(sampled_cam_matrixes[idx])
    for i in range(len(sorted_cam_matrix)):
        forward = sorted_cam_matrix[i].R[:3,2]
        forward_shift = forward * radius * (radius_scale - 1)
        sorted_cam_matrix[i].T += forward_shift
    cam_matrixes = []
    sorted_cam_matrix = [sorted_cam_matrix[0]] * 3 + sorted_cam_matrix + [sorted_cam_matrix[-1]] * 3
    z_height = sorted_cam_matrix[0].T[2] 
    for i in range(len(sorted_cam_matrix) - 3):
        q1 = R.from_matrix(sorted_cam_matrix[i].R).as_quat()
        q2 = R.from_matrix(sorted_cam_matrix[i+1].R).as_quat()
        q3 = R.from_matrix(sorted_cam_matrix[i+2].R).as_quat()
        q4 = R.from_matrix(sorted_cam_matrix[i+3].R).as_quat()
        t1 = sorted_cam_matrix[i].T
        t2 = sorted_cam_matrix[i+1].T
        t3 = sorted_cam_matrix[i+2].T
        t4 = sorted_cam_matrix[i+3].T
        num_per_frame = max(300 // (len(sorted_cam_matrix) - 3), 1)
        for j in range(num_per_frame):
            q = R.from_quat(spline_quat(j / num_per_frame, q1, q2, q3, q4))
            t = spline(j / num_per_frame, t1, t2, t3, t4)
            t[2] = z_height
            new_direction = center - t
            sorted_cam_matrix[i].R[:3,2] = new_direction / np.linalg.norm(new_direction)
            sorted_cam_matrix[i].R[:3,0] = np.cross(sorted_cam_matrix[i].R[:3,1], sorted_cam_matrix[i].R[:3,2])
            sorted_cam_matrix[i].R[:3,1] = np.cross(sorted_cam_matrix[i].R[:3,2], sorted_cam_matrix[i].R[:3,0])
            interpolation_info = Interpolation_Info(R = q.as_matrix(), T = t)
            cam_matrixes.append(interpolation_info)
    return cam_matrixes

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

### Sam added for interpolation rendering
class Interpolation_Info(NamedTuple):
    R: np.array
    T: np.array

def render_interpolations(dataset : ModelParams, iteration: int, pipeline: PipelineParams, skip_train : bool, skip_test : bool, fps = 30, duration = 5):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_interpolation(dataset.model_path, "train_interpolation", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, fps = fps, duration = duration)
def render_panoramas(dataset : ModelParams, iteration: int, pipeline: PipelineParams, skip_train : bool, skip_test : bool, fps = 30, duration = 2):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_panorama(dataset.model_path, "panorama", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, fps = fps, duration = duration)
def render_user_defined_videos(dataset : ModelParams, input_json , output_path ,iteration: int, pipeline: PipelineParams, skip_train : bool, skip_test : bool, fps = 30, duration = 2):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_user_defined(dataset.model_path, input_json , output_path ,"panorama", scene.loaded_iter,scene.getTrainCameras(), gaussians, pipeline, background, fps = fps, duration = duration)
def new_fov(fov,newpixel,oldpixel):
    import math
    return 2*math.atan(math.tan(0.5*fov)*newpixel/oldpixel)
    
def render_user_defined(model_path, input_json , output_path ,name, iteration,views, gaussians, pipeline, background, fps = 30, duration = 5):
    with open(input_json, 'r') as f:
        ref_transforms = json.load(f)

    camera_angle_x = ref_transforms['camera_angle_x'] if 'camera_angle_x' in ref_transforms else 0.785398
    camera_angle_y = ref_transforms['camera_angle_y'] if 'camera_angle_y' in ref_transforms else 0.785398
    # w, h = ref_transforms["w"],ref_transforms["h"]
    # fx = 0.5 * w / np.tan(0.5 * camera_angle_x)
    # fy = 0.5 * h / np.tan(0.5 * camera_angle_y)
    # focal = min(fx, fy)
    fovx = camera_angle_x
    fovy = camera_angle_y

    cam_matrixes = []
    for frame in ref_transforms['frames']:
        cam_matrixes.append(np.array(frame['transform_matrix']))

    cameras = ngptransform_to_gaussiancamera(cam_matrixes)
    video_images = []
    camera_points_set = []
    origin_points_set = []

    for idx, interpolation_info in enumerate(tqdm(cameras, desc="Rendering progress")):
        view_t = Camera(colmap_id=views[0].colmap_id,
                        R = interpolation_info.R,
                        T = interpolation_info.T,
                        FoVx = views[0].FoVx,
                        FoVy = views[0].FoVy,
                        image = views[0].original_image,
                        gt_alpha_mask = None,
                        image_name = views[0].image_name,
                        uid = views[0].uid,
                        data_device = views[0].data_device)
        # view_t.FoVx = new_fov(fovx,ref_transforms["w"],view_t.image_width)
        # view_t.FoVy = new_fov(fovy,ref_transforms["h"],view_t.image_height)
        # view_t.image_height = ref_transforms["h"]
        # view_t.image_width = ref_transforms["w"]
        rendering = render(view_t, gaussians, pipeline, background)["render"]
        rendering = np.array(rendering.cpu()).transpose(1, 2, 0).clip(0, 1)
        rendering_height, rendering_width = rendering.shape[:2]
        # 获取ref_transforms中的h和w值
        ref_h = ref_transforms["h"]
        ref_w = ref_transforms["w"]

        # 检查h和w是否大于rendering的高度和宽度
        if ref_h > rendering_height or ref_w > rendering_width:
            # 如果h或w大于rendering的对应尺寸，则使用原始大小
            cropped_rendering = rendering
        else:
            # 计算裁剪的起始和结束位置
            start_h = (rendering_height - ref_h) // 2
            end_h = start_h + ref_h
            start_w = (rendering_width - ref_w) // 2
            end_w = start_w + ref_w

            cropped_rendering = rendering[start_h:end_h, start_w:end_w]
        video_images.append(cropped_rendering)
        
        del view_t
    imageio.mimsave(output_path, video_images, fps = ref_transforms["fps"])
    output_image_path  = output_path.replace(".mp4", ".png")
    
    imageio.imwrite(output_image_path, (video_images[0] * 255).astype(np.uint8))
    
def render_interpolation(model_path, name, iteration, views, gaussians, pipeline, background, fps = 30, duration = 5):
    
    render_path = os.path.join(model_path)
    point_cloud_path = os.path.join(model_path, "point_cloud","iteration_30000","point_cloud.ply")
    interpolation_views,output_json,output_json_ios,cam_center = compute_panorama_path(views,300)#,fps*duration)
    print(os.path.join(render_path,"../interpolation_camera.json"))
    with open(os.path.join(render_path,"../interpolation_camera.json"), 'w') as f:
        json.dump(output_json, f,indent=4)
    with open(os.path.join(render_path,"../interpolation_camera_ios.json"), 'w') as f:
        json.dump(output_json_ios, f,indent=4)
    # save the center position
    with open(os.path.join(render_path,"../camera_center.json"), 'w') as f:
        json.dump({"center":cam_center.tolist()}, f,indent=4)
    makedirs(render_path, exist_ok=True)
    video_images = []

    for idx, interpolation_info in enumerate(tqdm(interpolation_views, desc="Rendering progress")):
        view_t = Camera(colmap_id=views[0].colmap_id,
                        R = interpolation_info.R,
                        T = interpolation_info.T,
                        FoVx = views[0].FoVx,
                        FoVy = views[0].FoVy,
                        image = views[0].original_image,
                        gt_alpha_mask = None,
                        image_name = views[0].image_name,
                        uid = views[0].uid,
                        data_device = views[0].data_device)
        
        rendering = render(view_t, gaussians, pipeline, background)["render"]
        video_images.append(np.array(rendering.cpu()).transpose(1, 2, 0).clip(0, 1))
        
        del view_t
    parent_dir = os.path.dirname(render_path)
    print(parent_dir)
    imageio.mimsave(os.path.join(parent_dir,"video.mp4"), video_images, fps = fps)
def render_panorama(model_path, name, iteration, views, gaussians, pipeline, background, fps = 30, duration = 2):
    point_cloud_path = os.path.join(model_path, "camera.ply")
    render_path = os.path.join(model_path)
    print("render is {}".format(render_path))
    print("here is the render view:")
    interpolation_views,_,_,_ = compute_panorama_path(views,60)
    parent_dir = os.path.dirname(render_path)
    # print(parent_dir)
    makedirs(render_path, exist_ok=True)
    makedirs(parent_dir,exist_ok=True)
    video_images = []
    # create a point cloud to record the camera position
    camera_position = []
    for idx, interpolation_info in enumerate(tqdm(interpolation_views, desc="Rendering progress")):
        view_t = Camera(colmap_id=views[0].colmap_id,
                        R = interpolation_info.R,
                        T = interpolation_info.T,
                        FoVx = views[0].FoVx,
                        FoVy = views[0].FoVy,
                        image = views[0].original_image,
                        gt_alpha_mask = None,
                        image_name = views[0].image_name,
                        uid = views[0].uid,
                        data_device = views[0].data_device)
        tmp = interpolation_info.T
        camera_position.append(tmp)
        
        rendering = render(view_t, gaussians, pipeline, background)["render"]
        video_images.append(np.flip(np.array(rendering.cpu()).transpose(1, 2, 0), 0))
        print(os.path.join(parent_dir,"panorama", '%d.png' % idx))
        os.makedirs(os.path.join(parent_dir,"panorama"),exist_ok=True)
        #clip image into 0,1
        rendering_normalized = np.array(rendering.cpu()).transpose(1, 2, 0).clip(0, 1)

        # Scale the normalized array to [0, 255] and convert to uint8
        rendering_uint8 = (rendering_normalized * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(parent_dir,"panorama", '%d.png' % idx), rendering_uint8)
    # save the camera position as a point cloud
    camera_position = np.array(camera_position)
    camera_position = camera_position.reshape(-1,3)
    o3d.io.write_point_cloud(point_cloud_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(camera_position)))
def slerp(quat1, quat2, t_values):
    quat1 = quat1 / np.linalg.norm(quat1)
    quat2 = quat2 / np.linalg.norm(quat2)
    cos_half_theta = np.dot(quat1, quat2)
    if abs(cos_half_theta) >= 1.0:
        return quat1
    helf_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
    if abs(sin_half_theta) < 0.001:
        result = (quat1 * 0.5 + quat2 * 0.5)
    else:
        ratio_a = np.sin((1 - t_values) * helf_theta) / sin_half_theta
        ratio_b = np.sin(t_values * helf_theta) / sin_half_theta
        result = (ratio_a[:, np.newaxis] * quat1 + ratio_b[:, np.newaxis] * quat2)
    return result

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--interpolation_train", action="store_true", help = "render interpolation of training")# sam added for interpolation rendering
    parser.add_argument("--fps", default=30, type=int, help = "fps of interpolation")# sam added for interpolation rendering
    parser.add_argument("--duration", default=5, type=int, help = "duration of interpolation")# sam added for interpolation rendering
    parser.add_argument("--user_render", action="store_true", help = "render user defined path") # add for user defined render path
    parser.add_argument("--input_user_defined_path",default="",type=str,help = "input user defined path")
    parser.add_argument("--output_user_defined_video",default="",type=str,help = "output user defined video")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.user_render: 
        render_user_defined_videos(model.extract(args),args.input_user_defined_path,args.output_user_defined_video, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, fps = 30, duration = 10)
    else:
        render_interpolations(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, fps = 30, duration = 10) # sam added for interpolation rendering
        render_panoramas(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, fps = 30, duration = 2)