"""
This code estimates 3D pose given model and sample point clouds as input using Open3D library.
It follows a global registration followed by refine registration (ICP) approach.

"""

from __future__ import division, print_function, unicode_literals  # To support both python 2 and python 3
import numpy as np
import math
from open3d import io, geometry, pipelines, visualization, utility


class pose3D(object):
    def __init__(self, model_orig_pcd, sample_pcd):
        """
        Arguments:
            model_orig_pcd: model point cloud
            sample_pcd: sample point cloud
        """
        self.model_pcd_original = model_orig_pcd
        self.XYZ_model = np.asarray(self.model_pcd_original.points)
        self.XYZ_min_model = np.min(self.XYZ_model, axis=0)
        self.XYZ_max_model = np.max(self.XYZ_model, axis=0)
        self.diameter_model = np.sqrt(np.square(self.XYZ_max_model[0] - self.XYZ_min_model[0]) +
                                      np.square(self.XYZ_max_model[1] - self.XYZ_min_model[1]) +
                                      np.square(self.XYZ_max_model[2] - self.XYZ_min_model[2]))
        self.model_pcd = geometry.PointCloud()  # To keep resized model points that can match the size of a sample
        # point cloud.

        self.sample_pcd = sample_pcd

    def prepare_dataset(self, model, sample, voxel_size):

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        sample_down = geometry.PointCloud.voxel_down_sample(sample, voxel_size)
        model_down = geometry.PointCloud.voxel_down_sample(model, voxel_size)

        radius_normal = voxel_size * 1  # 2 (default) 1 (new camera) 1.5 (old camera)
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        geometry.PointCloud.estimate_normals(sample_down, geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=10))  # 30 (default) 10 (new camera) 20 (old  camera)
        geometry.PointCloud.estimate_normals(model_down, geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=10))
        geometry.PointCloud.estimate_normals(sample, geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=10))
        geometry.PointCloud.estimate_normals(model, geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=10))

        radius_feature = voxel_size * 2  # 5 (default) 2 (new camera)  3(old camera)
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        source_fpfh = pipelines.registration.compute_fpfh_feature(sample_down,
                geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=40))  # 100 (default) 40 (new camera)
        # 60 (old camera)

        target_fpfh = pipelines.registration.compute_fpfh_feature(model_down,
                geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=40))
        return sample, model, sample_down, model_down, source_fpfh, target_fpfh

    def execute_global_registration(self, sample_down, model_down, sample_fpfh, model_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.0  # 1.5 (default) 1.0 (old camera)
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = pipelines.registration.registration_ransac_based_on_feature_matching(
            sample_down, model_down, sample_fpfh, model_fpfh, 0.075,
            pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.075)],
            pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        return result

    def execute_fast_global_registration(self, sample_down, model_down, sample_fpfh, model_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.2  # 0.5 (default) 0.2 (new camera)
        print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
        result = pipelines.registration.registration_fast_based_on_feature_matching(
            sample_down, model_down, sample_fpfh, model_fpfh,
            pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
        return result

    def refine_registration(self, sample, model, sample_fpfh, model_fpfh, voxel_size, result_global):
        distance_threshold = voxel_size * 0.2  # 0.4 (default) 0.2 (new camera) 0.3 (old camera)
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = pipelines.registration.registration_icp(sample, model,
                distance_threshold,
                result_global.transformation,              # result_ransac.transformation or result_fast.transformation
                pipelines.registration.TransformationEstimationPointToPlane())  # TransformationEstimationPointToPlane()
        # or TransformationEstimationPointToPoint()

        return result

    def combined_registration(self, sample, model):
            voxel_size = 0.001  # 0.001      #0.05 (default) # means 5cm for the dataset
            sample, model, sample_down, model_down, sample_fpfh, model_fpfh = self.prepare_dataset(model, sample,
                                                                                                   voxel_size)
            # result_global = self.execute_global_registration(sample_down, model_down, sample_fpfh, model_fpfh,
            # voxel_size) # Global using RANSAC
            result_global = self.execute_fast_global_registration(sample_down, model_down, sample_fpfh, model_fpfh,
                                                                  voxel_size)  # Fast global
            sample_down.transform(result_global.transformation)
            # print result_global
            # print 'Global T:', result_global.transformation
            result_icp = self.refine_registration(sample, model, sample_fpfh, model_fpfh, voxel_size, result_global)
            sample.transform(result_icp.transformation)
            # print result_icp
            # print 'ICP T:', result_icp.transformation
            rotation_matrix = result_icp.transformation[0:3, 0:3]
            # print rotation_matrix
            quaternion = self.quaternion_from_matrix(rotation_matrix, isprecise=False)
            # print 'Quaternion:', quaternion
            model.paint_uniform_color([1, 0, 0])  # Model is with green color
            sample.paint_uniform_color([0, 1, 0])
            return rotation_matrix, quaternion

    def quaternion_from_matrix(self, matrix, isprecise=False):
        """
        Return quaternion from rotation matrix.

        If isprecise is True, the input matrix is assumed to be a precise rotation matrix and a faster algorithm is
        used.

        """
        m = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4, ))
            t = np.trace(m)
            if t > m[3, 3]:
                q[0] = t
                q[3] = m[1, 0] - m[0, 1]
                q[2] = m[0, 2] - m[2, 0]
                q[1] = m[2, 1] - m[1, 2]
            else:
                i, j, k = 0, 1, 2
                if m[1, 1] > m[0, 0]:
                    i, j, k = 1, 2, 0
                if m[2, 2] > m[i, i]:
                    i, j, k = 2, 0, 1
                t = m[i, i] - (m[j, j] + m[k, k]) + m[3, 3]
                q[i] = t
                q[j] = m[i, j] + m[j, i]
                q[k] = m[k, i] + m[i, k]
                q[3] = m[k, j] - m[j, k]
                q = q[[3, 0, 1, 2]]
            q *= 0.5 / math.sqrt(t * m[3, 3])
        else:
            m00 = m[0, 0]
            m01 = m[0, 1]
            m02 = m[0, 2]
            m10 = m[1, 0]
            m11 = m[1, 1]
            m12 = m[1, 2]
            m20 = m[2, 0]
            m21 = m[2, 1]
            m22 = m[2, 2]
            # symmetric matrix k
            k = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                 [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                 [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                 [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            k /= 3.0
            # quaternion is eigenvector of k that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(k)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    def estimate_3Dpose(self):

        sample_pcd = self.sample_pcd
        XYZ_model = self.XYZ_model
        diameter_model = self.diameter_model
        model_pcd = self.model_pcd

        numpy_point_data = np.asarray(sample_pcd.points)
        print(numpy_point_data.shape)
        XYZ_min = np.min(numpy_point_data, axis=0)
        XYZ_max = np.max(numpy_point_data, axis=0)
        print('minimum and maximum: ', XYZ_min, XYZ_max)
        diameter_sample = np.sqrt(np.square(XYZ_max[0] - XYZ_min[0]) + np.square(XYZ_max[1] - XYZ_min[1]) +
                                  np.square(XYZ_max[2] - XYZ_min[2]))
        numpy_point_data = np.reshape(numpy_point_data, (-1, 3))
        sample_pcd.points = utility.Vector3dVector(numpy_point_data)
        # sample_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Flip it, otherwise the
        # pointcloud will be upside down
        sample_pcd.paint_uniform_color([0, 1, 0])

        ratio = diameter_sample / diameter_model  # Using diameter from pointcloud
        print('Ratio = mushroom_diameter/mushroom_diameter_model: ', ratio)
        XYZ_model_new = XYZ_model * ratio  # Resize the model points to match that of a sample
        model_pcd.points = utility.Vector3dVector(XYZ_model_new)
        # model_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Flip it, otherwise the
        # pointcloud will be upside down
        model_pcd.paint_uniform_color([1, 0, 0])
        visualization.draw_geometries([model_pcd + sample_pcd])
        rotation_matrix_estimated, quaternion_estimated = self.combined_registration(model_pcd, sample_pcd)
        print('rotation_matrix_estimated and rotation_vector_estimated: ', rotation_matrix_estimated)
        model_pcd.paint_uniform_color([1, 0, 0])
        sample_pcd.paint_uniform_color([0, 1, 0])
        visualization.draw_geometries([model_pcd + sample_pcd])


# Main function
def main():
    model_orig_pcd = io.read_point_cloud("model.pcd")
    sample_pcd = io.read_point_cloud("sample.pcd")  # Any sample point cloud can be given here
    Pose = pose3D(model_orig_pcd, sample_pcd)
    Pose.estimate_3Dpose()


# Execute from the interpreter
if __name__ == "__main__":
    main()
