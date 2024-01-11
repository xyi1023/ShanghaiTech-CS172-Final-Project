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
from tqdm import tqdm 
import json
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
def FlowDeformation(flow, xyz,camera,depth,canonical_depth):
    homogeneous_voxel_centers = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    camera_angle_X = camera["camera_angle_x"]
    fx = 0.5 * camera["w"]/ np.tan(0.5 * camera_angle_X)
    K = np.array([[fx,0,camera["cx"]],[0,fx,camera["cy"]],[0,0,1]])
    RT = np.array(camera["transform_matrix"]) #which is a 4*4 matrix
    RT[...,1]*=-1
    RT[...,2]*=-1
    RT = np.linalg.inv(RT)
    # convert it to 4*3 matrix
    RT = RT[:3,:]
    # to camera coordinate
    voxel_in_camera_coord = np.dot(RT, homogeneous_voxel_centers.T).T
    # to pixel coordinate
    voxel_in_pixel_coord = np.dot(K, voxel_in_camera_coord.T).T
    # normalize
    voxel_in_pixel_coord[:, 0] /= voxel_in_pixel_coord[:, 2]
    voxel_in_pixel_coord[:, 1] /= voxel_in_pixel_coord[:, 2]
    # round
    voxel_in_pixel_coord = np.round(voxel_in_pixel_coord).astype(int)
    x = voxel_in_pixel_coord[:,0]
    y = voxel_in_pixel_coord[:,1]
    valid_indices = (y < camera["h"]) & (x < camera["w"]) & (y >= 0) & (x >= 0)
    # print(valid_indices)
    dxyz = np.zeros((xyz.shape[0],3))
    answer = flow[voxel_in_pixel_coord[:, 1][valid_indices], voxel_in_pixel_coord[:, 0][valid_indices]]
    filtered_voxel_in_pixel_coord = voxel_in_pixel_coord[valid_indices]

    new_x = filtered_voxel_in_pixel_coord[:, 0] + answer[:, 0]
    new_y = filtered_voxel_in_pixel_coord[:, 1] + answer[:, 1]
    additional_valid_indices = (new_y < camera["h"]) & (new_x < camera["w"]) & (new_y >= 0) & (new_x >= 0)
    new_x = new_x[additional_valid_indices]
    new_y = new_y[additional_valid_indices]
    dz = depth[new_y, new_x] - canonical_depth[filtered_voxel_in_pixel_coord[:, 1][additional_valid_indices], filtered_voxel_in_pixel_coord[:, 0][additional_valid_indices]]
    dz = dz.cpu().numpy()
    answer = answer[additional_valid_indices]
    dx = answer[:,0] * dz
    dy = answer[:,1] * dz
    dvoxel_in_pixel_coord =np.linalg.inv(K) @ np.vstack([dx,dy,np.ones(new_x.shape[0])]) * dz
    dRT = np.array(camera["transform_matrix"]) #which is a 4*4 matrix
    dRT[...,1]*=-1
    dRT[...,2]*=-1
    dvoxel_in_camera_coord_homogeneous = np.vstack([dvoxel_in_pixel_coord, np.ones(new_x.shape[0])])
    dvoxel_in_world_coord = np.dot(dRT, dvoxel_in_camera_coord_homogeneous)
    dxyz_world = dvoxel_in_world_coord[:3, :].T
    dxyz[valid_indices][additional_valid_indices, :] = dxyz_world
    print(dxyz.shape)
    return dxyz,True


def training(dataset, opt, pipe, testing_iterations, saving_iterations,flow_folder):
    try:
        with open(f"{dataset.source_path}/dataset.json","r") as f:
            scene_infomation = json.load(f)
    except:
        scene_infomation = None
    with open("{}/transforms_train.json".format(dataset.source_path), "r") as f:
        cam_info = json.load(f)
    print(scene_infomation)
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    tracking_flow_dictionary = {}

    deform.train_setting(opt)

    scene = Scene(dataset, gaussians)
    # if Nerfies dataset(we adjust all dataset as the format of Nerfies dataset)
    
    if(flow_folder != ""):
        print("Read In Flow Information!")
        for i in range(len(scene_infomation["ids"])-1):
            flow_data = np.load("{}/{}_{}.npy".format(flow_folder,scene_infomation["ids"][i],scene_infomation["ids"][i+1]))
            tracking_flow_dictionary["{}_{}".format(i,i+1)] = flow_data
    init_data = np.load("{}/{}_{}.npy".format(flow_folder,scene_infomation["ids"][0],scene_infomation["ids"][1]))
    tracking_flow_list = [init_data for i in range(len(scene_infomation["ids"]))] 
    for i in range(1,len(scene_infomation["ids"])):
        for j in range(1,i):
            tracking_flow_list[i] = tracking_flow_list[i] + tracking_flow_dictionary["{}_{}".format(j,j+1)]
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    canonical_depth = scene.getTrainCameras().copy()[0].depth
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    Deformflow ={}
    for i in tqdm(range(len(scene.getTrainCameras()))):
        viewpoint_cam = scene.getTrainCameras().copy()[i]
        camera_information = {"w":cam_info["w"],"h":cam_info["h"],"cx":cam_info["cx"],"cy":cam_info["cy"],"camera_angle_x":cam_info["camera_angle_x"],"transform_matrix":cam_info["frames"][viewpoint_cam.uid]["transform_matrix"]}

        Deformflow[viewpoint_cam.uid] ,_= FlowDeformation(tracking_flow_list[viewpoint_cam.uid],gaussians.get_xyz.detach().cpu().numpy(),camera_information,viewpoint_cam.depth,canonical_depth)
    print("Flow calculate finish!")
    for iteration in range(1, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
        #                                                                                                        0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # print(viewpoint_stack[0].uid)
        # for i in range(len(viewpoint_stack)):
        #     print(viewpoint_stack[i].uid)
        # exit()
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            # print(len(cam_info["frames"]))
            # print(cam_info["frames"][viewpoint_cam.uid])
            # save the gaussian's points gaussians.get_xyz.detach().cpu().numpy()
            pre_cal_xyz = Deformflow[viewpoint_cam.uid]#(tracking_flow_list[viewpoint_cam.uid],gaussians.get_xyz.detach().cpu().numpy(),camera_information,viewpoint_cam.depth,canonical_depth)
            pre_cal_xyz_tensor = torch.from_numpy(pre_cal_xyz).to(gaussians.get_xyz.dtype)
            pre_cal_xyz_tensor = pre_cal_xyz_tensor.to("cuda")
            # print(pre_cal_xyz.shape)
            # def FlowDeformation(flow, xyz,camera,depth,canonical_depth):
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            # if(flag):
            warp_gaussian = pre_cal_xyz_tensor +gaussians.get_xyz
            # else:
                # warp_gaussian = gaussians.get_xyz
            # d_xyz, d_rotation, d_scaling = deform.step(warp_gaussian, time_input + ast_noise)
            d_xyz, d_rotation = deform.step(warp_gaussian, time_input + ast_noise)
            d_scaling = 0.0
            d_xyz = d_xyz + pre_cal_xyz_tensor
        # print("Iteration {} - Viewpoint {} - Time {}".format(iteration, viewpoint_cam.image_name, fid))
        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation = deform.step(xyz.detach(), time_input)
                    d_scaling =0.0
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--flow_folder",type=str,default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,args.flow_folder)

    # All done
    print("\nTraining complete.")
