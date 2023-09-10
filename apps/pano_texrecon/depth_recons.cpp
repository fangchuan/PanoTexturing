////
//// Created by yuandong on 2022/2/24.

#include "depth_recons.h"

#include <open3d/geometry/BoundingVolume.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/pipelines/registration/ColoredICP.h>
#include <util/alignment.h>

#include <opencv2/core/core.hpp>

#include "base/cubemap.h"
#include "floor_plan/grid_map.h"
#include "log.h"
#include "mesh/pcl_utils.h"
#include "mesh/utils.h"
#include "mobile/depth_estimation/layout_depth.h"
#include "mobile/utils.h"
#include "mve/mesh_info.h"
#include "open3d/core/EigenConverter.h"
#include "open3d/t/geometry/RaycastingScene.h"
#include "poisson_recon_interface.h"
#include "util/timer.h"

namespace lyj {

DepthRecons::DepthRecons(const DepthReconsOptions &options,
                         const Database *database)
    : options_(options), database_(database) {
  settings_.keep_unseen_faces = true;
  settings_.tone_mapping = tex::ToneMapping::TONE_MAPPING_GAMMA;
  settings_.data_term = tex::DataTerm::DATA_TERM_AREA;
  settings_.geometric_visibility_test = true;
  settings_.outlier_removal = tex::OutlierRemoval::OUTLIER_REMOVAL_NONE;
}

bool DepthRecons::Build(const std::vector<FloorPlan> &rooms,
                        ErrorMessage &msg) {
  //  options_.obj_filepath = options_.output_dir + "/model.obj";

  if (!options_.obj_filepath.empty()) {
    if (!IsFileExist(options_.obj_filepath)) {
      AWARN << "Can not Find " << options_.obj_filepath << std::endl;
      // EditErrorMessage(ERR_NO_FILE, "Can not Find " + options_.obj_filepath,
      //                  msg);
      return false;
    }
    AINFO << "Find " << options_.obj_filepath << "  ..." << std::endl;
    return TextureRemapping(rooms, msg);
  }

  if (options_.use_layout) {
    return BuildWithLayout(rooms, msg);
  } else {
    return BuildWithoutLayout(rooms, msg);
  }
  return true;
}

bool DepthRecons::TextureRemapping(const std::vector<FloorPlan> &rooms,
                                   ErrorMessage &msg) {
  // read obj file
  open3d::geometry::TriangleMesh mesh;
  AINFO << "Reading " << options_.obj_filepath << "  ..." << std::endl;

  // open3d::io::ReadTriangleMeshOptions read_triangle_mesh_options;
  // open3d::io::ReadTriangleMeshFromOBJ(options_.obj_filepath, mesh,
  // read_triangle_mesh_options);
  open3d::io::ReadTriangleMeshFromOBJ(options_.obj_filepath, mesh, false);

  if (!GetRoomsInfo(rooms, *database_, mesh_rooms_info_)) {
    return false;
  }

  double camera_height = mesh_rooms_info_[0].camera_height;
  for (int i = 0; i < mesh.vertices_.size(); ++i) {
    mesh.vertices_[i][1] = mesh.vertices_[i][1] - camera_height;
  }

  // 使用修模后的模型
  mesh_rooms_info_[0].mesh_3d = mesh;

  // 纹理映射及保存
  options_.whole_building_mapping = true;
  AINFO << "Running Texture Mapping ...";
  TextureMapping(rooms);

  return true;
}

bool DepthRecons::BuildWithoutLayout(const std::vector<FloorPlan> &rooms,
                                     ErrorMessage &msg) {
  AINFO << "Build model without layout" << std::endl;

  // 得到normal-pc
  AINFO << "Running Merge Rooms Norm Point Cloud.";
  if (!MergeRoomsNormalPC(rooms, msg)) {
    return false;
  }

  // 柏松重建
  AINFO << "Running Poisson Surface Reconstruction.";
  if (!PoissonSurfaceReconstruction()) {
    msg.message = "Failed to do poisson surface reconstruction";
    msg.success = false;
    return false;
  }

  //  if (options_.simplify_mesh) {
  //    AINFO << "Running Simplify Mesh.";
  //    SimplifyMesh();
  //  }

  // mesh优化
  AINFO << "Running Mesh Optimization.";
  if (options_.remove_tiny_mesh) {
    for (int i = 0; i < mesh_rooms_info_.size(); ++i) {
      RemoveTinyMesh(mesh_rooms_info_[i].mesh_3d);
    }
  }

  // 纹理映射及保存
  options_.whole_building_mapping = true;
  AINFO << "Running Texture Mapping.";
  TextureMapping(rooms);

  return true;
}

bool DepthRecons::BuildWithLayout(const std::vector<FloorPlan> &rooms,
                                  ErrorMessage &msg) {
  if (!GetRoomsInfo(rooms, *database_, mesh_rooms_info_)) {
    return false;
  }

  // 得到normal-pc
  AINFO << "Running Generating Rooms Norm Point Cloud.";
  if (!GenerateRoomsNormalPC(rooms, msg)) {
    return false;
  }

  // 柏松重建
  AINFO << "Running Poisson Surface Reconstruction.";
  if (!PoissonSurfaceReconstruction()) {
    msg.message = "Failed to do poisson surface reconstruction";
    msg.success = false;
    return false;
  }

  // 门扣除
  AINFO << "Running Croping Room Doors.";
  CropRoomDoorsMesh(rooms);

  if (options_.simplify_mesh) {
    AINFO << "Running Simplify Mesh.";
    SimplifyMesh();
  }

  // mesh优化
  AINFO << "Running Mesh Optimization.";
  if (options_.remove_tiny_mesh) {
    for (int i = 0; i < mesh_rooms_info_.size(); ++i) {
      RemoveTinyMesh(mesh_rooms_info_[i].mesh_3d);
    }
  }

  // 把在地面下的点拉回地面
  if (options_.align_floor) {
    AlignFloorMesh();
  }

  // 纹理映射及保存
  AINFO << "Running Texture Mapping.";
  TextureMapping(rooms);

  return true;
}

mve::TriangleMesh::Ptr DepthRecons::ConvertOpen3DMeshToMVEMesh(
    const open3d::geometry::TriangleMesh &open3d_mesh) {
  mve::TriangleMesh::Ptr mve_mesh = mve::TriangleMesh::create();
  mve::TriangleMesh::VertexList &vertices = mve_mesh->get_vertices();
  mve::TriangleMesh::FaceList &faces = mve_mesh->get_faces();

  for (int i = 0; i < open3d_mesh.vertices_.size(); ++i) {
    math::Vec3f vertex(0.0f, 0.0f, 0.0f);
    vertex[0] = open3d_mesh.vertices_[i](0);
    vertex[1] = open3d_mesh.vertices_[i](1);
    vertex[2] = open3d_mesh.vertices_[i](2);
    vertices.emplace_back(vertex);
  }

  for (int i = 0; i < open3d_mesh.triangles_.size(); ++i) {
    faces.emplace_back(open3d_mesh.triangles_[i](0));
    faces.emplace_back(open3d_mesh.triangles_[i](1));
    faces.emplace_back(open3d_mesh.triangles_[i](2));
  }

  return mve_mesh;
}

bool DepthRecons::GetRoomsInfo(const std::vector<FloorPlan> &rooms,
                               const Database &database,
                               std::vector<MeshRoomInfo> &mesh_rooms_info) {
  mesh_rooms_info.clear();

  for (int i = 0; i < rooms.size(); ++i) {
    MeshRoomInfo mesh_room;

    double room_height_total = 0;
    double camera_height_total = 0;
    mesh_room.room_name = rooms[i].GetName();
    mesh_room.main_index = SelectedMainShot(rooms[i]);

    int count = 0;
    for (int j = 0; j < rooms[i].NumShots(); ++j) {
      std::string shot_name = rooms[i].GetShot(j).GetName();
      if (!database.HasLayout(shot_name)) {
        AWARN << "Does not find shot in database.";
        return false;
      }
      const Layout layout = database.GetLayout(shot_name);
      const Panorama panorama = database.GetPano(shot_name);
      camera_height_total =
          camera_height_total + std::fabs(panorama.GetCameraHeight());
      room_height_total = room_height_total + std::fabs(layout.GetRoomHeight());
      count++;
    }
    CHECK_NE(count, 0);
    mesh_room.camera_height = camera_height_total / count;
    mesh_room.room_height = room_height_total / count;

    mesh_rooms_info.emplace_back(mesh_room);
  }
  return true;
}

bool DepthRecons::GenerateRoomsNormalPC(const std::vector<FloorPlan> &rooms,
                                        ErrorMessage &msg) {
  std::vector<FloorPlan> rooms_update = rooms;
  for (int i = 0; i < rooms_update.size(); ++i) {
    auto room = rooms_update[i];
    std::string room_name = room.GetName();
    open3d::geometry::PointCloud per_room_norm_pc, room_layout_pc_open3d;

    if (options_.shots_position_refinement) {
      if (!ShotsPositionRefinement(room)) {
        AWARN << "Failed to refine shots position: " << room.GetName();
      }
    }

    GenerateRoomLayoutPC(room, mesh_rooms_info_[i].camera_height,
                         mesh_rooms_info_[i].room_height,
                         room_layout_pc_open3d);

    for (int j = 0; j < room.NumShots(); ++j) {
      Shot shot = room.GetShot(j);
      std::string shot_name = shot.GetName();
      AINFO << "Processing room: " << room_name << " shot: " << shot_name
            << " index: " << j << std::endl;
      const cv::Mat shot_pano = database_->GetPano(shot_name).GetImage();

      // 判断是否存在depth
      if (!database_->HasDepth(shot_name)) {
        AWARN << "Fail to get pano depth: " << shot_name << std::endl;
        return false;
      }
      const auto shot_mono_depth = database_->GetDepth(shot_name);

      // layout-depth与mono-depth进行融合
      LayoutDepth gen_layout_depth;
      if (!gen_layout_depth.GenerateLayoutDepth(room, j, database_)) {
        AWARN << "Failed to generate layout depth" << std::endl;
        return false;
      }
      auto shot_layout_depth = gen_layout_depth.GetDepth();

      MergeDepth gen_merge_depth(merge_options_);
      gen_merge_depth.SetColorImage(shot_pano);
      gen_merge_depth.MergeProcess(shot_mono_depth, shot_layout_depth, room, j,
                                   gen_layout_depth.GetFloorMask(),
                                   gen_layout_depth.GetWallPlaneMask());
      auto shot_merged_depth = gen_merge_depth.GetMergedDepth();

      cv::Mat mask_depth;
      auto mask = gen_merge_depth.GetMonoDepthMask();
      cv::bitwise_and(shot_merged_depth, shot_merged_depth, mask_depth, mask);

      Eigen::Matrix4d inv_T = shot.GetT().inverse();

      if (mesh_rooms_info_[i].main_index == j) {
        open3d::geometry::PointCloud temp_normpc;
        for (int k = 0; k < room_layout_pc_open3d.points_.size(); ++k) {
          Eigen::Vector3d pc;
          pc << room_layout_pc_open3d.points_[k][0],
              room_layout_pc_open3d.points_[k][1],
              room_layout_pc_open3d.points_[k][2];

          auto pc_equi = (inv_T * pc.homogeneous()).hnormalized();
          Eigen::Vector2d corrd2d = XYZ2UV(pc_equi, mask);

          if (corrd2d[0] < 0 || corrd2d[1] < 0 ||
              corrd2d[1] >= mask.size().height ||
              corrd2d[0] >= mask.size().width) {
            continue;
          }
          bool masked_flag = mask.at<uchar>(corrd2d[1], corrd2d[0]);

          if (!masked_flag) {
            temp_normpc.points_.emplace_back(room_layout_pc_open3d.points_[k]);
            temp_normpc.colors_.emplace_back(room_layout_pc_open3d.colors_[k]);
            temp_normpc.normals_.emplace_back(
                room_layout_pc_open3d.normals_[k]);
            continue;
          }

          if (!Visable(room, shot, pc)) {
            temp_normpc.points_.emplace_back(room_layout_pc_open3d.points_[k]);
            temp_normpc.colors_.emplace_back(room_layout_pc_open3d.colors_[k]);
            temp_normpc.normals_.emplace_back(
                room_layout_pc_open3d.normals_[k]);
            continue;
          }
        }
        room_layout_pc_open3d = temp_normpc;
      }

      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          shot_cloud(0);
      std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
          shot_color(0);
      PanoDepth2PointCloud(mask_depth, shot_pano, 0.01,
                           options_.max_preserved_depth, shot_cloud, shot_color,
                           options_.max_point_set_height, 0,
                           shot_merged_depth.size());

      Transform2WorldSpace(shot_cloud, shot.GetT());

      open3d::geometry::PointCloud furniture_pc;
      for (int k = 0; k < shot_cloud.size(); ++k) {
        furniture_pc.points_.emplace_back(shot_cloud[k]);
        furniture_pc.colors_.emplace_back(Eigen::Vector3d(
            shot_color[k](0), shot_color[k](1), shot_color[k](2)));
      }
      // cv::imshow("layout-depth",shot_layout_depth/10);
      // cv::imshow("mono-depth",shot_mono_depth/10);
      // cv::imshow("mask",mask);
      // cv::waitKey(0);

      // 因为norm估计为30个最近领点，所以需要大于30
      if (furniture_pc.points_.size() > 30) {
        furniture_pc.EstimateNormals();
        furniture_pc.OrientNormalsTowardsCameraLocation(shot.GetPosition());

        // 每个shot删除离群点
        // std::shared_ptr<open3d::geometry::PointCloud> removed =
        //   std::shared_ptr<open3d::geometry::PointCloud>(nullptr);
        // std::tie(removed,std::ignore) =
        // furniture_pc.RemoveRadiusOutliers(4,0.1); furniture_pc = *removed;
        for (int k = 0; k < furniture_pc.points_.size(); ++k) {
          per_room_norm_pc.points_.emplace_back(furniture_pc.points_[k]);
          per_room_norm_pc.colors_.emplace_back(furniture_pc.colors_[k]);
          per_room_norm_pc.normals_.emplace_back(furniture_pc.normals_[k]);
        }

        // 是否进行降采样
        //        if (options_.pc_down_sample) {
        //          per_room_norm_pc =
        //              *per_room_norm_pc.VoxelDownSample(options_.furniture_resolution);
        //        }
      }
    }
    // vis
    //  open3d::geometry::PointCloud show_temp = per_room_norm_pc;
    //  show_temp.PaintUniformColor({255,0,0});
    //  open3d::visualization::DrawGeometries(
    //      {std::make_shared<open3d::geometry::PointCloud>(show_temp)},
    //      "room cloud");
    // point filter

    RoomLumpsFilter(per_room_norm_pc, room, -mesh_rooms_info_[i].camera_height,
                    options_.height_low_threshold,
                    mesh_rooms_info_[i].room_height * 0.95);
    // vis
    //  per_room_norm_pc.PaintUniformColor({0,0,255});
    //  open3d::visualization::DrawGeometries(
    //      {std::make_shared<open3d::geometry::PointCloud>(per_room_norm_pc)},
    //      "room cloud filtered");

    // 整个房间离群点删除
    // std::shared_ptr<open3d::geometry::PointCloud> removed =
    // std::shared_ptr<open3d::geometry::PointCloud>(nullptr);
    // std::tie(removed,std::ignore) =
    // per_room_norm_pc.RemoveRadiusOutliers(4,0.1);
    // removed->PaintUniformColor({0,255,0});

    // open3d::visualization::DrawGeometries(
    //     {std::make_shared<open3d::geometry::PointCloud>(show_temp),
    //     std::make_shared<open3d::geometry::PointCloud>(per_room_norm_pc),removed},
    //     "room cloud filtered");

    // per_room_norm_pc = *removed;

    for (int j = 0; j < room_layout_pc_open3d.points_.size(); ++j) {
      per_room_norm_pc.points_.emplace_back(room_layout_pc_open3d.points_[j]);
      per_room_norm_pc.colors_.emplace_back(room_layout_pc_open3d.colors_[j]);
      per_room_norm_pc.normals_.emplace_back(room_layout_pc_open3d.normals_[j]);
    }

    auto room_doors = room.GetDoors();
    for (int j = 0; j < room_doors.size(); ++j) {
      if (room_doors[j].valid) {
        AddDoorEdges(room_doors[j], mesh_rooms_info_[i].camera_height,
                     per_room_norm_pc);
      }
    }
    //    VisOpen3dPC(per_room_norm_pc);
    mesh_rooms_info_[i].norm_pc_3d = per_room_norm_pc;
  }
  return true;
}
bool DepthRecons::RoomLumpsFilter(open3d::geometry::PointCloud &point_cloud,
                                  const FloorPlan &room, double floor_height,
                                  double low_threshold, double high_threshold) {
  int ratio = 100, padding = 5;
  int shot_r_int = static_cast<int>(options_.shot_region_radius * ratio);
  // project point to bottom plane
  auto bound = point_cloud.GetAxisAlignedBoundingBox();

  double length = abs(bound.max_bound_(0) - bound.min_bound_(0));
  double height = abs(bound.max_bound_(2) - bound.min_bound_(2));
  int plane_cols = static_cast<int>(length * ratio) + 2 * padding;
  int plane_rows = static_cast<int>(height * ratio) + 2 * padding;
  cv::Mat bottom_plane_valid = cv::Mat::zeros(plane_rows, plane_cols, CV_8UC1);

  for (size_t i = 0; i < point_cloud.points_.size(); ++i) {
    auto point_temp = point_cloud.points_[i];
    double point_height = point_temp(1) - floor_height;
    if (point_height > low_threshold && point_height < high_threshold) {
      int col =
          static_cast<int>(abs(point_temp(0) - bound.min_bound_(0)) * ratio) +
          padding;
      int row =
          static_cast<int>(abs(point_temp(2) - bound.min_bound_(2)) * ratio) +
          padding;
      bottom_plane_valid.at<uchar>(row, col) = 255;
    }
  }

  cv::Mat shots_region = cv::Mat::zeros(plane_rows, plane_cols, CV_8UC1);
  for (auto single_shot : room.GetShots()) {
    Eigen::Vector3d position = single_shot.GetPosition();
    int col = static_cast<int>(abs(position(0) - bound.min_bound_(0)) * ratio) +
              padding;
    int row = static_cast<int>(abs(position(2) - bound.min_bound_(2)) * ratio) +
              padding;
    cv::circle(shots_region, cv::Point(col, row), shot_r_int,
               cv::Scalar(255, 255, 255), -1);
  }
  cv::Mat elment =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(padding, padding));
  cv::dilate(bottom_plane_valid, bottom_plane_valid, elment);
  cv::erode(bottom_plane_valid, bottom_plane_valid, elment);

  // cv::imshow("mask before shots",bottom_plane_valid);
  cv::bitwise_and(bottom_plane_valid, ~shots_region, bottom_plane_valid);
  // cv::imshow("mask after shots",bottom_plane_valid);
  // cv::imshow("mask shots",shots_region);
  // cv::waitKey(0);
  // get valid region and filter region
  for (size_t i = 0; i < point_cloud.points_.size(); ++i) {
    auto point_temp = point_cloud.points_[i];
    int col =
        static_cast<int>(abs(point_temp(0) - bound.min_bound_(0)) * ratio) +
        padding;
    int row =
        static_cast<int>(abs(point_temp(2) - bound.min_bound_(2)) * ratio) +
        padding;
    if (bottom_plane_valid.at<uchar>(row, col) < 128) {
      point_cloud.points_[i](1) = floor_height;
    }
  }
  return true;
}

// 541行
//       SaveVectEigenAsPly(shot_cloud,options_.output_dir+"/"+shot_name+".ply");
// std::vector<float> dat2d_temp;
// auto T = shot.GetT();
// for (int k = 0; k < 4; ++k) {
// for (int l = 0; l < 4; ++l) {
// dat2d_temp.emplace_back(T(k,l));
// }
// }
//
// nlohmann::json pose_json;
// pose_json["pose"] = dat2d_temp;
////       写入json
// std::string output_json_name =
//     options_.output_dir + "/" + shot_name + ".json";
// std::ofstream fout(output_json_name);
// fout << pose_json.dump(4) << std::endl;

bool DepthRecons::OptimizePointSet(const open3d::geometry::PointCloud &room_pc,
                                   const Eigen::Vector3d &orientation_reference,
                                   open3d::geometry::PointCloud &shot_pc) {
  lyj::PoissonRecon poisson_recon_lyj;
  poisson_recon_lyj.SetInput(room_pc);
  poisson_recon_lyj.performReconstruction();
  auto room_mesh = poisson_recon_lyj.GetOutput();

  open3d::t::geometry::RaycastingScene current_scene;
  auto cube = open3d::t::geometry::TriangleMesh::FromLegacy(room_mesh);
  current_scene.AddTriangles(cube);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix;
  matrix.resize(shot_pc.points_.size(), 6);

  for (int i = 0; i < shot_pc.points_.size(); ++i) {
    Eigen::Vector3d point_dir = shot_pc.points_[i] - orientation_reference;
    point_dir = point_dir.normalized();
    matrix.row(i) << orientation_reference[0], orientation_reference[1],
        orientation_reference[2], point_dir[0], point_dir[1], point_dir[2];
  }

  auto ray = open3d::core::eigen_converter::EigenMatrixToTensor(matrix);
  auto ans = current_scene.CastRays(ray);

  std::vector<Eigen::Vector3d> primitive_normals;
  float *t_hit;
  for (auto iter = ans.begin(); iter != ans.end(); iter++) {
    if (iter->first == "primitive_normals") {
      primitive_normals =
          open3d::core::eigen_converter::TensorToEigenVector3dVector(
              iter->second);
    }
    if (iter->first == "t_hit") {
      t_hit = iter->second.GetDataPtr<float>();
    }
  }

  float tol_low = 0.1;
  float tol_up = 0.3;

  for (int i = 0; i < shot_pc.points_.size(); ++i) {
    if (t_hit[i] > 5) continue;
    // norm相反
    auto norm_dir = primitive_normals[i].dot(shot_pc.normals_[i]);
    double cosangle = std::acos(norm_dir) / M_PI * 180;
    if (cosangle < 90) continue;
    auto ori_hit_dis = (shot_pc.points_[i] - orientation_reference).norm();
    // && ori_hit_dis<(t_hit[i]+tol_up)
    if (ori_hit_dis > (t_hit[i] - tol_low)) {
      float move_dist = std::min(0.3, (ori_hit_dis - t_hit[i] + 0.15));
      Eigen::Vector3d update_point =
          shot_pc.points_[i] + move_dist * shot_pc.normals_[i];
      shot_pc.points_[i] = update_point;
    }
  }

  return true;
}

bool DepthRecons::MergeRoomsNormalPC(const std::vector<FloorPlan> &rooms,
                                     ErrorMessage &msg) {
  mesh_rooms_info_.clear();
  MeshRoomInfo merged_room_info;

  for (int i = 0; i < rooms.size(); ++i) {
    auto room = rooms[i];
    for (int j = 0; j < rooms[i].NumShots(); ++j) {
      Shot shot = room.GetShot(j);
      std::string shot_name = shot.GetName();
      AINFO << "Merge room: " << room.GetName() << " shot: " << shot_name
            << " index: " << j << std::endl;
      const cv::Mat shot_pano = database_->GetPano(shot_name).GetImage();

      if (!database_->HasDepth(shot_name)) {
        AWARN << "Fail to get pano depth: " << shot_name << std::endl;
        return false;
      }
      // @TODO 把rooms.json里的scale拿出来 把这个 scale 放到database里面?
      const auto shot_mono_depth = database_->GetDepth(shot_name);

      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
          shot_cloud(0);
      std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
          shot_color(0);
      PanoDepth2PointCloud(shot_mono_depth, shot_pano, 0.01,
                           options_.max_preserved_depth, shot_cloud, shot_color,
                           options_.max_point_set_height, 0.15,
                           shot_mono_depth.size());
      Transform2WorldSpace(shot_cloud, shot.GetT());

      open3d::geometry::PointCloud shot_pc;
      for (int k = 0; k < shot_cloud.size(); ++k) {
        shot_pc.points_.emplace_back(shot_cloud[k]);
        shot_pc.colors_.emplace_back(Eigen::Vector3d(
            shot_color[k](0), shot_color[k](1), shot_color[k](2)));
      }

      shot_pc.EstimateNormals();
      shot_pc.OrientNormalsTowardsCameraLocation(shot.GetPosition());

      //      if(merged_room_info.norm_pc_3d.points_.size()>100){
      //        OptimizePointSet(merged_room_info.norm_pc_3d,
      //        shot.GetPosition(),shot_pc);
      //      }
      //      SaveNormalOpen3dAsPly(shot_pc,
      //      options_.output_dir+"/"+shot_name+".ply");

      for (int k = 0; k < shot_pc.points_.size(); ++k) {
        merged_room_info.norm_pc_3d.points_.emplace_back(shot_pc.points_[k]);
        merged_room_info.norm_pc_3d.colors_.emplace_back(shot_pc.colors_[k]);
        merged_room_info.norm_pc_3d.normals_.emplace_back(shot_pc.normals_[k]);
      }
    }
  }

  //  open3d::io::ReadPointCloudOption option;
  //  open3d::io::ReadPointCloudFromPLY("/Users/yuandong/Documents/Git_project_DAMO/mesh_recons_eval_dataset/merged_pcd.ply",merged_room_info.norm_pc_3d,option);
  // 是否进行降采样
  if (options_.pc_down_sample) {
    merged_room_info.norm_pc_3d = *merged_room_info.norm_pc_3d.VoxelDownSample(
        options_.furniture_resolution);
  }

  //  SaveNormalOpen3dAsPly(merged_room_info.norm_pc_3d,
  //  options_.output_dir+"/"+"merged_pc.ply");

  mesh_rooms_info_.emplace_back(merged_room_info);
  return true;
}

bool DepthRecons::ShotsPositionRefinement(FloorPlan &room) {
  std::unordered_map<int, open3d::geometry::PointCloud> shots_pointset;
  std::unordered_map<int, open3d::geometry::PointCloud> ori_pointset;

  if (room.NumShots() < 2) return true;

  for (int i = 0; i < room.NumShots(); ++i) {
    Shot shot = room.GetShot(i);
    std::string shot_name = shot.GetName();
    const cv::Mat pano_img = database_->GetPano(shot_name).GetImage();
    cv::Mat mono_depth = database_->GetDepth(shot_name);

    LayoutDepth gen_layout_depth;
    if (!gen_layout_depth.GenerateLayoutDepth(room, i, database_)) {
      AWARN << "Failed to generate layout depth" << std::endl;
      return false;
    }
    auto layout_depth = gen_layout_depth.GetDepth();
    merge_options_.method = MergeMethod::MaskOuterArea;
    MergeDepth gen_merge_depth(merge_options_);
    gen_merge_depth.MergeProcess(mono_depth, layout_depth,
                                 gen_layout_depth.GetFloorMask());
    mono_depth = gen_merge_depth.GetMergedDepth();

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        shot_cloud(0);
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>
        shot_color(0);
    PanoDepth2PointCloud(mono_depth, pano_img, 0.01,
                         options_.max_preserved_depth, shot_cloud, shot_color,
                         options_.max_point_set_height, 0, mono_depth.size());

    open3d::geometry::PointCloud pc;
    for (int k = 0; k < shot_cloud.size(); ++k) {
      pc.points_.emplace_back(shot_cloud[k]);
      pc.colors_.emplace_back(Eigen::Vector3d(
          shot_color[k](0), shot_color[k](1), shot_color[k](2)));
    }
    //    ori_pointset[i] = *pc.VoxelDownSample(0.05);
    pc.Transform(shot.GetT());
    pc.EstimateNormals();
    pc.OrientNormalsTowardsCameraLocation(shot.GetPosition());

    shots_pointset[i] = *pc.VoxelDownSample(0.05);
  }

  Timer timer;
  timer.Start();
  std::vector<int> updated_index;
  updated_index.emplace_back(0);

  std::vector<Shot> updated_shots;
  updated_shots.resize(shots_pointset.size());
  updated_shots[0] = room.GetShot(0);

  while (updated_index.size() != shots_pointset.size()) {
    int source_index, target_index;
    SelectClosedShotsPair(room, updated_index, source_index, target_index);
    auto source = shots_pointset[source_index];
    auto target = shots_pointset[target_index];

    auto result = open3d::pipelines::registration::RegistrationICP(
        source, target, 0.2, Eigen::Matrix4d::Identity(),
        open3d::pipelines::registration::TransformationEstimationPointToPoint(
            false),
        open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6,
                                                                30));

    double angle_error =
        std::abs(std::acos(
            fmin(fmax(((result.transformation_.block<3, 3>(0, 0).transpose() *
                        Eigen::Matrix3d::Identity())
                           .trace() -
                       1) /
                          2,
                      -1.0),
                 1.0))) /
        M_PI * 180;
    AINFO << "The registration angle error between "
          << room.GetShot(source_index).GetName() << " and "
          << room.GetShot(target_index).GetName() << " :" << angle_error;

    Shot source_shot = room.GetShot(source_index);
    Shot target_shot = room.GetShot(target_index);

    if (angle_error < 5.0 || result.transformation_(2, 3) > 0.1) {
      Eigen::Matrix4d new_trans = updated_shots[target_index].GetT() *
                                  target_shot.GetT().inverse() *
                                  result.transformation_ * source_shot.GetT();
      source_shot.UpdateT(new_trans);
      updated_shots[source_index] = source_shot;
    } else {
      AWARN << "Registration failed or bad shot";
      Eigen::Matrix4d new_trans =
          updated_shots[target_index].GetT() * target_shot.GetT().inverse() *
          Eigen::Matrix4d::Identity() * source_shot.GetT();
      source_shot.UpdateT(new_trans);
      updated_shots[source_index] = source_shot;
    }

    //    if (shots_pointset.size()<5){
    //      auto source_vis = ori_pointset[source_index];
    //      auto source_ori = ori_pointset[source_index];
    //      auto target_vis = ori_pointset[target_index];
    //      VisOpen3dTwoPointCloud(source_ori.Transform(room.GetShot(source_index).GetT()),
    //                             source_vis.Transform(updated_shots[source_index].GetT()),
    //                             target_vis.Transform(updated_shots[target_index].GetT()));
    //    }
  }
  std::string print_string = "Registration Room :" + room.GetName();
  timer.End(print_string.c_str());

  room.SetShots(updated_shots);

  return true;
}

void DepthRecons::RemoveTinyMesh(open3d::geometry::TriangleMesh &input_mesh) {
  std::vector<int> clusters;
  std::vector<size_t> cluster_n_triangles;
  std::vector<double> cluster_area;
  std::tie(clusters, cluster_n_triangles, cluster_area) =
      input_mesh.ClusterConnectedTriangles();

  if (cluster_n_triangles.size() == 1) return;
  std::vector<size_t> indices;

  for (int i = 0; i < clusters.size(); ++i) {
    if (clusters[i] == 0 ||
        cluster_n_triangles[clusters[i]] > options_.num_triangles) {
      for (int j = 0; j < 3; ++j) {
        indices.emplace_back(input_mesh.triangles_[i][j]);
      }
    }
  }
  std::vector<size_t>::iterator ite = unique(indices.begin(), indices.end());
  indices.erase(ite, indices.end());
  input_mesh = *input_mesh.SelectByIndex(indices);
}

void DepthRecons::GenerateRoomLayoutPC(
    const FloorPlan &room, double camera_height, double room_height,
    open3d::geometry::PointCloud &room_layout_pc) {
  room_layout_pc.Clear();
  std::string shot0_name = room.GetShot(0).GetName();

  for (int i = 0; i < room.NumWalls(); ++i) {
    auto wall = room.GetWall(i);
    Eigen::Vector3d right = wall.GetRightEnd();
    Eigen::Vector3d left = wall.GetLeftEnd();
    double sample_num = (right - left).norm() / options_.layout_resolution;
    for (int j = 0; j < sample_num; ++j) {
      double lamda = j / sample_num;
      auto sample = (lamda)*right + (1 - lamda) * left;
      Eigen::Vector3d from, end;
      from << sample(0), -std::fabs(camera_height), sample(2);
      end << sample(0), std::fabs(room_height) - std::fabs(camera_height),
          sample(2);
      AddSamplePoints(from, end, wall.GetNormal(), room_layout_pc,
                      room_height / options_.layout_resolution);
    }
  }

  GridMap::Options options;
  options.resolution = options_.layout_resolution;
  MapRange map_range = CalcMapRange(room, options.resolution, options.pad);
  GridMap gm(map_range, options);
  gm.AddFloorPlan(room);
  cv::Mat mask = gm.GetMat();
  for (size_t y = 0; y < mask.rows; y++) {
    for (size_t x = 0; x < mask.cols; x++) {
      if (mask.at<uchar>(y, x) > 10) {
        Eigen::Vector3d sample = gm.UV2XYZ(cv::Point(x, y));
        Eigen::Vector3d floor_sample =
            Eigen::Vector3d(sample(0), -std::fabs(camera_height), sample(2));
        room_layout_pc.points_.emplace_back(floor_sample);
        room_layout_pc.colors_.emplace_back(Eigen::Vector3d(255, 0, 0));
        room_layout_pc.normals_.emplace_back(Eigen::Vector3d(0, 1, 0));
        Eigen::Vector3d ceilling_sample = Eigen::Vector3d(
            sample(0), room_height - std::fabs(camera_height), sample(2));
        room_layout_pc.points_.emplace_back(ceilling_sample);
        room_layout_pc.colors_.emplace_back(Eigen::Vector3d(255, 0, 0));
        room_layout_pc.normals_.emplace_back(Eigen::Vector3d(0, -1, 0));
      }
    }
  }
}

void DepthRecons::AddDoorEdges(const fp::Door &door, double camera_height,
                               open3d::geometry::PointCloud &norm_pc) {
  Eigen::Vector3d upper_left, upper_right, down_left, down_right;
  upper_left << door.left(0), door.height - std::fabs(camera_height),
      door.left(2);
  upper_right << door.right(0), door.height - std::fabs(camera_height),
      door.right(2);
  down_left << door.left(0), -std::fabs(camera_height), door.left(2);
  down_right << door.right(0), -std::fabs(camera_height), door.right(2);
  AddSamplePoints(upper_left, upper_right, door.normal, norm_pc);
  AddSamplePoints(down_left, down_right, door.normal, norm_pc);
  AddSamplePoints(upper_left, down_left, door.normal, norm_pc);
  AddSamplePoints(down_right, upper_right, door.normal, norm_pc);

  double sample_num = (upper_left - upper_right).norm() / 0.01;
  for (int j = 0; j < sample_num; ++j) {
    double lamda = j / sample_num;
    auto sample = (lamda)*upper_left + (1 - lamda) * upper_right;
    Eigen::Vector3d from, end;
    from << sample(0), -std::fabs(camera_height), sample(2);
    end << sample(0), door.height - std::fabs(camera_height), sample(2);
    AddSamplePoints(from, end, door.normal, norm_pc, sample_num);
  }
}

void DepthRecons::AddSamplePoints(const Eigen::Vector3d &from,
                                  const Eigen::Vector3d &to,
                                  const Eigen::Vector3d &norm,
                                  open3d::geometry::PointCloud &xyz_norm,
                                  double sample_num) {
  for (int i = 0; i < sample_num; ++i) {
    double lamda = i / sample_num;
    auto sample = (lamda)*from + (1 - lamda) * to;
    xyz_norm.points_.emplace_back(sample);
    xyz_norm.normals_.emplace_back(norm);
    xyz_norm.colors_.emplace_back(Eigen::Vector3d(255, 0, 0));
  }
}

bool DepthRecons::PoissonSurfaceReconstruction() {
  if (mesh_rooms_info_.empty()) {
    AWARN << "The room size is 0" << std::endl;
    return false;
  }

  for (int i = 0; i < mesh_rooms_info_.size(); i++) {
    AINFO << "Surface reconstruction room: " << mesh_rooms_info_[i].room_name
          << std::endl;

    lyj::PoissonRecon poisson_recon_lyj;
    poisson_recon_lyj.SetInput(mesh_rooms_info_[i].norm_pc_3d);
    poisson_recon_lyj.performReconstruction();
    mesh_rooms_info_[i].mesh_3d = poisson_recon_lyj.GetOutput();
  }

  return true;
}

void DepthRecons::CropRoomDoorsMesh(const std::vector<FloorPlan> &rooms) {
  for (int i = 0; i < mesh_rooms_info_.size(); i++) {
    auto room_doors = rooms[i].GetDoors();
    double camera_height = mesh_rooms_info_[i].camera_height;
    for (int j = 0; j < room_doors.size(); ++j) {
      if (room_doors[j].valid) {
        Eigen::Vector3f crop_min, crop_max;
        auto door = room_doors[j];
        crop_min << std::min(door.right(0), door.left(0)) -
                        0.1 * std::fabs(door.normal(0)),
            -std::fabs(camera_height) + 0.1,
            std::min(door.right(2), door.left(2)) -
                0.1 * std::fabs(door.normal(2));
        crop_max << std::max(door.right(0), door.left(0)) +
                        0.1 * std::fabs(door.normal(0)),
            door.height - std::fabs(camera_height),
            std::max(door.right(2), door.left(2)) +
                0.1 * std::fabs(door.normal(2));
        std::vector<size_t> indices;
        for (int k = 0; k < mesh_rooms_info_[i].mesh_3d.vertices_.size(); ++k) {
          bool low_flag =
              (mesh_rooms_info_[i].mesh_3d.vertices_[k][0] > crop_min[0] &&
               mesh_rooms_info_[i].mesh_3d.vertices_[k][1] > crop_min[1] &&
               mesh_rooms_info_[i].mesh_3d.vertices_[k][2] > crop_min[2]);
          bool up_flag =
              (mesh_rooms_info_[i].mesh_3d.vertices_[k][0] < crop_max[0] &&
               mesh_rooms_info_[i].mesh_3d.vertices_[k][1] < crop_max[1] &&
               mesh_rooms_info_[i].mesh_3d.vertices_[k][2] < crop_max[2]);
          if (!low_flag || !up_flag) {
            indices.emplace_back(k);
          }
        }
        mesh_rooms_info_[i].mesh_3d =
            *mesh_rooms_info_[i].mesh_3d.SelectByIndex(indices);
      }
    }
  }
}

void DepthRecons::AlignFloorMesh() {
  for (int j = 0; j < mesh_rooms_info_.size(); j++) {
    double camera_height = mesh_rooms_info_[j].camera_height;
    for (int i = 0; i < mesh_rooms_info_[j].mesh_3d.vertices_.size(); ++i) {
      if (mesh_rooms_info_[j].mesh_3d.vertices_[i][1] <
          -std::fabs(camera_height)) {
        mesh_rooms_info_[j].mesh_3d.vertices_[i][1] = -std::fabs(camera_height);
      }
    }
  }
}

bool DepthRecons::TextureMapping(const std::vector<FloorPlan> &rooms) {
  if (options_.whole_building_mapping) {
    WholeBuildingTextureMapping(rooms);
  } else {
    PerRoomTextureMapping(rooms);
  }

  return true;
}

void DepthRecons::WholeBuildingTextureMapping(
    const std::vector<FloorPlan> &rooms) {
  tex::TexturePatches building_texture_patches;
  mve::TriangleMesh::Ptr building_texture_mesh = mve::TriangleMesh::create();

  AINFO << "Texture mapping room: " << mesh_rooms_info_[0].room_name
        << std::endl;

  auto per_mve_mesh = ConvertOpen3DMeshToMVEMesh(mesh_rooms_info_[0].mesh_3d);

  tex::TextureViews per_texture_views, pano_views;

  for (int index = 0; index < rooms.size(); ++index) {
    auto room = rooms[index];
    for (int j = 0; j < room.NumShots(); ++j) {
      AddTextureViewsFromPano(*database_, room.GetShot(j),
                              options_.texture_bottom_mask, per_texture_views);
    }
  }

  if (options_.whole_building_unseen_fill) {
    for (int index = 0; index < rooms.size(); ++index) {
      auto room = rooms[index];
      for (int j = 0; j < room.NumShots(); ++j) {
        const cv::Mat main_pano =
            database_->GetPano(room.GetShot(j).GetName()).GetImage();
        cv::Mat blur_pano;
        cv::GaussianBlur(main_pano, blur_pano, cv::Size(15, 15), 15, 15, 4);
        Eigen::Matrix4d T = room.GetShot(j).GetT().inverse();
        mve::CameraInfo camera_info;
        camera_info.rot[0] = T(0, 0);
        camera_info.rot[1] = T(0, 1);
        camera_info.rot[2] = T(0, 2);
        camera_info.rot[3] = T(1, 0);
        camera_info.rot[4] = T(1, 1);
        camera_info.rot[5] = T(1, 2);
        camera_info.rot[6] = T(2, 0);
        camera_info.rot[7] = T(2, 1);
        camera_info.rot[8] = T(2, 2);
        camera_info.trans[0] = T(0, 3);
        camera_info.trans[1] = T(1, 3);
        camera_info.trans[2] = T(2, 3);
        tex::TextureView texture_view(0, camera_info, blur_pano);
        pano_views.emplace_back(texture_view);
      }
    }
  }

  tex::TexturePatches per_texture_patches;
  GenerateMVETexturePatches(per_mve_mesh, per_texture_views, pano_views,
                            per_texture_patches);

  MergeTextureMesh(per_texture_patches, building_texture_patches, per_mve_mesh,
                   building_texture_mesh,
                   Eigen::Vector3d(0, mesh_rooms_info_[0].camera_height, 0));

  SaveBuildingTextureMesh(building_texture_mesh, building_texture_patches);
}

void DepthRecons::PerRoomTextureMapping(const std::vector<FloorPlan> &rooms) {
  tex::TexturePatches building_texture_patches;
  mve::TriangleMesh::Ptr building_texture_mesh = mve::TriangleMesh::create();

  for (int index = 0; index < mesh_rooms_info_.size(); ++index) {
    AINFO << "Texture mapping room: " << mesh_rooms_info_[index].room_name
          << std::endl;
    auto per_mve_mesh =
        ConvertOpen3DMeshToMVEMesh(mesh_rooms_info_[index].mesh_3d);

    tex::TextureViews per_texture_views, pano_views;

    auto room = rooms[index];
    for (int j = 0; j < room.NumShots(); ++j) {
      AddTextureViewsFromPano(*database_, room.GetShot(j),
                              options_.texture_bottom_mask, per_texture_views);
    }

    {
      int main_index = mesh_rooms_info_[index].main_index;
      const cv::Mat main_pano =
          database_->GetPano(room.GetShot(main_index).GetName()).GetImage();
      cv::Mat blur_pano;
      cv::GaussianBlur(main_pano, blur_pano, cv::Size(15, 15), 15, 15, 4);
      Eigen::Matrix4d T = room.GetShot(main_index).GetT().inverse();
      mve::CameraInfo camera_info;
      camera_info.rot[0] = T(0, 0);
      camera_info.rot[1] = T(0, 1);
      camera_info.rot[2] = T(0, 2);
      camera_info.rot[3] = T(1, 0);
      camera_info.rot[4] = T(1, 1);
      camera_info.rot[5] = T(1, 2);
      camera_info.rot[6] = T(2, 0);
      camera_info.rot[7] = T(2, 1);
      camera_info.rot[8] = T(2, 2);
      camera_info.trans[0] = T(0, 3);
      camera_info.trans[1] = T(1, 3);
      camera_info.trans[2] = T(2, 3);
      tex::TextureView texture_view(0, camera_info, blur_pano);
      pano_views.emplace_back(texture_view);
    }

    tex::TexturePatches per_texture_patches;
    GenerateMVETexturePatches(per_mve_mesh, per_texture_views, pano_views,
                              per_texture_patches);

    MergeTextureMesh(
        per_texture_patches, building_texture_patches, per_mve_mesh,
        building_texture_mesh,
        Eigen::Vector3d(0, mesh_rooms_info_[index].camera_height, 0));
  }

  SaveBuildingTextureMesh(building_texture_mesh, building_texture_patches);
}

void DepthRecons::SaveBuildingTextureMesh(
    const mve::TriangleMesh::Ptr &mesh, tex::TexturePatches &texture_patches) {
  tex::TextureAtlases texture_atlases;
  tex::generate_texture_atlases(&texture_patches, settings_, &texture_atlases);

  AINFO << "Building obj model:" << options_.building_name << std::endl;
  tex::Model model;
  tex::build_model(mesh, texture_atlases, &model);

  if (!IfFolderNotExistThenCreate(options_.output_dir)) {
    AWARN << "Can not creat path: " << options_.output_dir;
  }

  std::string output_mesh = options_.output_dir + "/" + options_.building_name;
  tex::Model::save(model, output_mesh);
  AINFO << "Finished building obj model." << std::endl;
}

void DepthRecons::GenerateMVETexturePatches(
    const mve::TriangleMesh::Ptr &mve_mesh, tex::TextureViews &texture_views,
    const tex::TextureViews &pano_views, tex::TexturePatches &texture_patches) {
  mve::MeshInfo mesh_info(mve_mesh);
  tex::prepare_mesh(&mesh_info, mve_mesh);

  std::size_t const num_faces = mve_mesh->get_faces().size() / 3;
  tex::Graph graph(num_faces);
  tex::build_adjacency_graph(mve_mesh, mesh_info, &graph);

  tex::DataCosts data_costs(num_faces, texture_views.size());
  tex::calculate_data_costs(mve_mesh, &texture_views, settings_, &data_costs);

  tex::view_selection(data_costs, &graph, settings_);

  tex::VertexProjectionInfos vertex_projection_infos;

  if (pano_views.size() < 1) {
    tex::generate_texture_patches(graph, mve_mesh, mesh_info, &texture_views,
                                  settings_, &vertex_projection_infos,
                                  &texture_patches);
    tex::global_seam_leveling(graph, mve_mesh, mesh_info,
                              vertex_projection_infos, &texture_patches);
    tex::local_seam_leveling(graph, mve_mesh, vertex_projection_infos,
                             &texture_patches);
  } else {
    tex::TexturePatches unseen_texture_patches;
    tex::generate_texture_patches(graph, mve_mesh, mesh_info, &texture_views,
                                  settings_, &vertex_projection_infos,
                                  &texture_patches, pano_views,
                                  &unseen_texture_patches);
    tex::global_seam_leveling(graph, mve_mesh, mesh_info,
                              vertex_projection_infos, &texture_patches);
    tex::local_seam_leveling(graph, mve_mesh, vertex_projection_infos,
                             &texture_patches);
    texture_patches.insert(texture_patches.end(),
                           unseen_texture_patches.begin(),
                           unseen_texture_patches.end());
  }
}

void DepthRecons::MergeTextureMesh(const tex::TexturePatches &source_patches,
                                   tex::TexturePatches &target_patches,
                                   const mve::TriangleMesh::Ptr &source_mesh,
                                   mve::TriangleMesh::Ptr &target_mesh,
                                   const Eigen::Vector3d &mesh_shift) {
  // merge texture-patch
  for (int i = 0; i < source_patches.size(); ++i) {
    for (int j = 0; j < source_patches[i]->get_faces().size(); ++j) {
      source_patches[i]->get_faces()[j] = source_patches[i]->get_faces()[j] +
                                          target_mesh->get_faces().size() / 3;
    }
  }
  target_patches.insert(target_patches.begin(), source_patches.begin(),
                        source_patches.end());
  // merge mesh
  int offset = target_mesh->get_vertices().size();
  for (int i = 0; i < source_mesh->get_vertices().size(); ++i) {
    math::Vec3f new_vert;
    new_vert[0] = source_mesh->get_vertices()[i][0] + mesh_shift[0];
    new_vert[1] = source_mesh->get_vertices()[i][1] + mesh_shift[1];
    new_vert[2] = source_mesh->get_vertices()[i][2] + mesh_shift[2];
    target_mesh->get_vertices().push_back(new_vert);
  }

  target_mesh->get_vertex_normals().insert(
      target_mesh->get_vertex_normals().end(),
      source_mesh->get_vertex_normals().begin(),
      source_mesh->get_vertex_normals().end());
  target_mesh->get_face_normals().insert(
      target_mesh->get_face_normals().end(),
      source_mesh->get_face_normals().begin(),
      source_mesh->get_face_normals().end());

  for (int i = 0; i < source_mesh->get_faces().size(); ++i) {
    unsigned int face_index = source_mesh->get_faces()[i] + offset;
    target_mesh->get_faces().emplace_back(face_index);
  }
}

void DepthRecons::SimplifyMesh() {
  for (int i = 0; i < mesh_rooms_info_.size(); ++i) {
    auto mesh_3d = mesh_rooms_info_[i].mesh_3d;
    if (options_.simplify_voxel_size < 0) continue;
    auto simplified_mesh =
        mesh_3d.SimplifyVertexClustering(options_.simplify_voxel_size);
    AINFO << mesh_rooms_info_[i].room_name
          << " faces before simplify: " << mesh_3d.triangles_.size()
          << " after: " << simplified_mesh->triangles_.size();
    mesh_rooms_info_[i].mesh_3d = *simplified_mesh;
  }
}

int DepthRecons::SelectedMainShot(const FloorPlan &room) {
  GridMap::Options options;
  MapRange map_range = CalcMapRange(room, options.resolution, options.pad);
  GridMap gm(map_range, options);
  gm.AddFloorPlan(room);
  cv::Point centroid = gm.GetCentroid();

  double max_ratio = 0;
  double min_dis = 1e5;
  int main_shot = 0;
  int all_area = cv::countNonZero(gm.GetMat());

  for (int i = 0; i < room.NumShots(); ++i) {
    cv::Point spot = gm.XYZ2UV(room.GetShot(i).GetPosition());
    double dis = std::sqrt((spot.x - centroid.x) * (spot.x - centroid.x) +
                           (spot.y - centroid.y) * (spot.y - centroid.y));

    int vis_area = cv::countNonZero(gm.VisibleArea(spot));
    double ratio = double(vis_area) / double(all_area + 0.01);

    if (ratio > max_ratio) {
      main_shot = i;
      max_ratio = ratio;
      min_dis = dis;
    } else if (std::fabs(ratio - max_ratio) < 0.005) {
      if (dis < min_dis) {
        main_shot = i;
        max_ratio = ratio;
        min_dis = dis;
      }
    }
  }
  return main_shot;
}

std::vector<cv::Mat> DepthRecons::GenerateTextureImages(
    const cv::Mat &pano_img) {
  lyj::Cubemap cubemap;
  cubemap.GenerateFromPano(pano_img, options_.cubemap_resolution,
                           options_.cubemap_fov);
  std::vector<cv::Mat> texture_images;
  texture_images.resize(6);
  texture_images[0] = RotateImage(cubemap.Top(), -90);
  texture_images[1] = RotateImage(cubemap.Bottom(), 90);
  texture_images[2] = FlipX(FlipY(cubemap.Back()));
  texture_images[3] = FlipX(FlipY(cubemap.Front()));
  texture_images[4] = FlipX(FlipY(cubemap.Right()));
  texture_images[5] = FlipX(FlipY(cubemap.Left()));
  return texture_images;
}

void DepthRecons::Transform2WorldSpace(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        &shot_cloud,
    const Eigen::Matrix4d &T) {
  for (int i = 0; i < shot_cloud.size(); ++i) {
    shot_cloud[i] = (T * shot_cloud[i].homogeneous()).hnormalized();
  }
}

// @todo
void DepthRecons::SetUserRoomHeight(const double &room_height) {
  options_.user_room_height = room_height;
}

}  // namespace lyj
