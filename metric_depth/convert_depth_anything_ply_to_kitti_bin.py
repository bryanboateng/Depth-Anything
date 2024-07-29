import argparse
import os.path
import pathlib

import open3d as o3d
import numpy as np


def save_to_binary(point_cloud, intensity, output_file_path: pathlib.Path):
    points = np.asarray(point_cloud.points)
    intensity = intensity.reshape(-1, 1)
    point_cloud_data = np.hstack((points, intensity)).astype(np.float32)
    point_cloud_data.tofile(str(output_file_path))


def convert_to_greyscale(pcd):
    colors = np.asarray(pcd.colors)
    intensity = np.mean(colors, axis=1)
    greyscale_colors = np.tile(intensity[:, None], (1, 3))
    pcd.colors = o3d.utility.Vector3dVector(greyscale_colors)
    return pcd, intensity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plyfile", type=pathlib.Path)
    args = parser.parse_args()
    ply_file_path: pathlib.Path = args.plyfile

    point_cloud = o3d.io.read_point_cloud(str(ply_file_path))
    rotation_matrix = point_cloud.get_rotation_matrix_from_xyz((np.deg2rad(-90), 0, 0))
    point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    point_cloud, intensity = convert_to_greyscale(point_cloud)
    save_to_binary(point_cloud, intensity, ply_file_path.with_suffix(".bin"))


if __name__ == "__main__":
    main()
