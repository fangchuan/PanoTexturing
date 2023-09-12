////
//// Created by yuandong on 2022/2/24.
////
//
#ifndef LYJ_RECONSTRUCTION_DEV_SRC_MESH_DEPTH_RECONS_H_
#define LYJ_RECONSTRUCTION_DEV_SRC_MESH_DEPTH_RECONS_H_

#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>

#include <opencv2/core/core.hpp>

#include "mve/mesh.h"
#include "tex/texturing.h"

namespace lyj {

enum TextureMappingMethod {
  /** map texture to mesh*/
  MainImage = 0,  //@todo
  MultiImage = 1
};

struct MeshRoomInfo {
  std::string room_name;
  double camera_height = 1.6;
  double room_height = 1;
  int main_index = 0;
  open3d::geometry::PointCloud norm_pc_3d;
  open3d::geometry::TriangleMesh mesh_3d;
};

struct PanoReconsOptions {
  // output path
  std::string output_dir;

  std::string building_name = "model";

  double user_room_height = -1;

  // layout-constrain
  bool use_layout = true;

  // point-cloud
  double max_point_set_height = 10.0;
  double max_preserved_depth = 8.0;
  bool pc_down_sample = false;
  bool shots_position_refinement = false;

  // save
  bool save_rooms = false;

  // texture
  bool texture_mapping_method = TextureMappingMethod::MultiImage;
  bool whole_building_mapping = false;
  bool texture_bottom_mask =
      true;  // 贴图的时候是否把底部中心区域进行valid-mask
  int cubemap_resolution = 600;
  double cubemap_fov = (100.0 / 180.0) * M_PI;
  bool whole_building_unseen_fill = true;

  // 修模后obj文件路径
  std::string obj_filepath = "";

  // mesh
  double furniture_resolution = 0.05;
  double layout_resolution = 0.05;
  bool remove_tiny_mesh = true;
  bool align_floor = false;
  int num_triangles = 1000;
  bool simplify_mesh = true;
  double simplify_voxel_size = 0.05;

  // lumps
  double height_low_threshold = 0.4;
  double shot_region_radius = 0.5;
};

class PanoRecons {
 public:
  PanoRecons(const PanoReconsOptions &options,
             const std::string &input_img_path);

  bool Build();

  void SetUserRoomHeight(const double &room_height);

 private:
  // // @brief：使用墙线（layout信息）进行约束
  // bool BuildWithLayout(const std::vector<FloorPlan> &rooms, ErrorMessage
  // &msg);

  // // @brief：不使用墙线（无layout信息）
  // bool BuildWithoutLayout(const std::vector<FloorPlan> &rooms,
  //                         ErrorMessage &msg);

  bool TextureRemapping();

  bool TextureMapping(const cv::Mat &pano_img);

  void SaveBuildingTextureMesh(const mve::TriangleMesh::Ptr &mesh,
                               tex::TexturePatches &texture_patches);

  void Transform2WorldSpace(
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          &shot_cloud,
      const Eigen::Matrix4d &T);

  // 对mesh进行处理
  void MergeTextureMesh(const tex::TexturePatches &source_patches,
                        tex::TexturePatches &target_patches,
                        const mve::TriangleMesh::Ptr &source_mesh,
                        mve::TriangleMesh::Ptr &target_mesh,
                        const Eigen::Vector3d &mesh_shift = Eigen::Vector3d(0,
                                                                            0,
                                                                            0));

  void WholeBuildingTextureMapping(const cv::Mat &input_pano);

  // void PerRoomTextureMapping(const cv::Mat &input_pano);

  void RemoveTinyMesh(open3d::geometry::TriangleMesh &input_mesh);

  std::vector<cv::Mat> GenerateTextureImages(const cv::Mat &pano_img);

  void GenerateMVETexturePatches(const mve::TriangleMesh::Ptr &mve_mesh,
                                 tex::TextureViews &texture_views,
                                 const tex::TextureViews &pano_views,
                                 tex::TexturePatches &texture_patches);

  mve::TriangleMesh::Ptr ConvertOpen3DMeshToMVEMesh(
      const open3d::geometry::TriangleMesh &open3d_mesh);

  void SimplifyMesh();

  PanoReconsOptions options_;
  std::vector<MeshRoomInfo> mesh_rooms_info_;

  tex::Settings settings_;

  // input panorama file path
  std::string input_pano_filepath_;
};

}  // namespace lyj

#endif  // LYJ_RECONSTRUCTION_DEV_SRC_MESH_DEPTH_RECONS_H_
