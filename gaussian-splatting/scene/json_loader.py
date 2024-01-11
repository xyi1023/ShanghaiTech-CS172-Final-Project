# hongyu add json_loader.py 
import numpy as np
import collections
import struct
import xml.etree.ElementTree as ET
import os 
import json
from pathlib import Path

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

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

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_json_extrinsics(path):
    images = {}
    # get type of images
    images_path = os.path.join(os.path.dirname(path), "images")
    # use the first image to decide the type of images
    image_file = os.listdir(images_path)[0]
    image_type = image_file.split(".")[-1]
    
    with open(path, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            image_id = frame["file_path"].split("/")[-1].split(".")[0]
            camera_id = frame["file_path"].split("/")[-1].split(".")[0]
            image_name = str(image_id) + "." + image_type
            
            flip_mat = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            matrix = np.matmul(np.array(frame["transform_matrix"]), flip_mat)
            matrix = np.linalg.inv(matrix)
            R = matrix[:3, :3]
            T = matrix[:3, 3]
            qvec = -rotmat2qvec(R)
            tvec = T.reshape([1,3])
            
            xys = None
            point3D_ids = None
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_json_intrinsics(path):
    cameras = {}
    with open(path, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        width = contents["w"]
        height = contents["h"]
        for idx, frame in enumerate(frames):          
            camera_id = frame["file_path"].split("/")[-1].split(".")[0]
            params = np.array([frame["fl_x"],frame["fl_y"]])
            model = "PINHOLE"
            cameras[camera_id] = Camera(id=camera_id, model=model,
                                        width=width, height=height,
                                        params=params)
            with open("jyh.txt", "a") as f:
                f.write(str(cameras[camera_id]) + "\n")
    return cameras

# hongyu add json_loader.py end