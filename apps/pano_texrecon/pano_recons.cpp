////
//// Created by yuandong on 2022/2/24.

#include "pano_recons.h"

#include <mve/mesh_io_ply.h>
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/pipelines/registration/ColoredICP.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cubemap.h"
#include "log.h"
#include "mve/mesh_info.h"
#include "open3d/core/EigenConverter.h"
#include "tex/texturing.h"
#include "util/timer.h"
#include "utils.h"

namespace lyj {

PanoRecons::PanoRecons(const PanoReconsOptions &options,
                       const std::string &input_pano_filepath)
    : options_(options) {
  input_pano_filepath_ = input_pano_filepath;

  settings_.keep_unseen_faces = true;
  settings_.tone_mapping = tex::ToneMapping::TONE_MAPPING_GAMMA;
  settings_.data_term = tex::DataTerm::DATA_TERM_AREA;
  settings_.geometric_visibility_test = true;
  settings_.outlier_removal = tex::OutlierRemoval::OUTLIER_REMOVAL_NONE;
}

bool PanoRecons::Build() {
  assert(!input_pano_filepath_.empty());
  assert(options_.obj_filepath.empty());

  if (!options_.obj_filepath.empty()) {
    if (!IsFileExist(options_.obj_filepath)) {
      AWARN << "Can not Find " << options_.obj_filepath << std::endl;
      return false;
    }
    AINFO << "Find " << options_.obj_filepath << "  ..." << std::endl;
    return TextureRemapping();
  }

  return true;
}

bool PanoRecons::TextureRemapping() {
  // read obj file
  open3d::geometry::TriangleMesh mesh;
  AINFO << "Reading " << options_.obj_filepath << "  ..." << std::endl;

  open3d::io::ReadTriangleMeshOptions read_triangle_mesh_options;
  open3d::io::ReadTriangleMeshFromPLY(options_.obj_filepath, mesh,
                                      read_triangle_mesh_options);

  MeshRoomInfo room_info;
  room_info.camera_height = 0.0;
  mesh_rooms_info_.emplace_back(room_info);

  // 使用修模后的模型
  mesh_rooms_info_[0].mesh_3d = mesh;

  if (options_.simplify_mesh) {
    AINFO << "Running Simplify Mesh.";
    SimplifyMesh();
  }

  // mesh优化
  if (options_.remove_tiny_mesh) {
    AINFO << "Running Mesh Optimization.";
    for (int i = 0; i < mesh_rooms_info_.size(); ++i) {
      RemoveTinyMesh(mesh_rooms_info_[i].mesh_3d);
    }
  }

  // read input panorama
  cv::Mat input_pano = cv::imread(input_pano_filepath_, cv::IMREAD_UNCHANGED);
  // 纹理映射及保存
  options_.whole_building_mapping = true;
  AINFO << "Running Texture Mapping ...";
  TextureMapping(input_pano);

  return true;
}

mve::TriangleMesh::Ptr PanoRecons::ConvertOpen3DMeshToMVEMesh(
    const open3d::geometry::TriangleMesh &open3d_mesh) {
  mve::TriangleMesh::Ptr mve_mesh = mve::TriangleMesh::create();
  mve::TriangleMesh::VertexList &vertices = mve_mesh->get_vertices();
  mve::TriangleMesh::NormalList &normals = mve_mesh->get_vertex_normals();
  mve::TriangleMesh::FaceList &faces = mve_mesh->get_faces();
  mve::TriangleMesh::NormalList &face_normals = mve_mesh->get_face_normals();

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

void PanoRecons::RemoveTinyMesh(open3d::geometry::TriangleMesh &input_mesh) {
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

bool PanoRecons::TextureMapping(const cv::Mat &input_pano) {
  if (options_.whole_building_mapping) {
    WholeBuildingTextureMapping(input_pano);
  } else {
    // PerRoomTextureMapping(input_pano);
  }

  return true;
}

void PanoRecons::WholeBuildingTextureMapping(const cv::Mat &input_pano) {
  tex::TexturePatches building_texture_patches;
  mve::TriangleMesh::Ptr building_texture_mesh = mve::TriangleMesh::create();

  AINFO << "Texture mapping room: " << mesh_rooms_info_[0].room_name
        << std::endl;

  auto per_mve_mesh = ConvertOpen3DMeshToMVEMesh(mesh_rooms_info_[0].mesh_3d);

  tex::TextureViews per_texture_views, pano_views;

  AddTextureViewsFromPano(input_pano, options_.texture_bottom_mask,
                          per_texture_views);

  if (options_.whole_building_unseen_fill) {
    for (int i = 0; i < 1; ++i) {
      cv::Mat blur_pano;
      cv::GaussianBlur(input_pano, blur_pano, cv::Size(15, 15), 15, 15, 4);
      // Eigen::Matrix4d T = room.GetShot(j).GetT().inverse();
      Eigen::Matrix4d T_pano = Eigen::Matrix4d::Identity();
      T_pano.block<3, 1>(0, 3) = Eigen::Vector3d(0, 1.6, 0);
      Eigen::Matrix4d T = T_pano.inverse();
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

  AINFO << "GenerateMVETexturePatches ..." << std::endl;
  tex::TexturePatches per_texture_patches;
  GenerateMVETexturePatches(per_mve_mesh, per_texture_views, pano_views,
                            per_texture_patches);

  AINFO << "MergeTextureMesh ..." << std::endl;
  MergeTextureMesh(per_texture_patches, building_texture_patches, per_mve_mesh,
                   building_texture_mesh,
                   Eigen::Vector3d(0, mesh_rooms_info_[0].camera_height, 0));

  AINFO << "SaveBuildingTextureMesh ..." << std::endl;
  SaveBuildingTextureMesh(building_texture_mesh, building_texture_patches);
}

// void PanoRecons::PerRoomTextureMapping(const std::vector<FloorPlan> &rooms) {
//   tex::TexturePatches building_texture_patches;
//   mve::TriangleMesh::Ptr building_texture_mesh = mve::TriangleMesh::create();

//   for (int index = 0; index < mesh_rooms_info_.size(); ++index) {
//     AINFO << "Texture mapping room: " << mesh_rooms_info_[index].room_name
//           << std::endl;
//     auto per_mve_mesh =
//         ConvertOpen3DMeshToMVEMesh(mesh_rooms_info_[index].mesh_3d);

//     tex::TextureViews per_texture_views, pano_views;

//     auto room = rooms[index];
//     for (int j = 0; j < room.NumShots(); ++j) {
//       AddTextureViewsFromPano(*database_, room.GetShot(j),
//                               options_.texture_bottom_mask,
//                               per_texture_views);
//     }

//     {
//       int main_index = mesh_rooms_info_[index].main_index;
//       const cv::Mat main_pano =
//           database_->GetPano(room.GetShot(main_index).GetName()).GetImage();
//       cv::Mat blur_pano;
//       cv::GaussianBlur(main_pano, blur_pano, cv::Size(15, 15), 15, 15, 4);
//       Eigen::Matrix4d T = room.GetShot(main_index).GetT().inverse();
//       mve::CameraInfo camera_info;
//       camera_info.rot[0] = T(0, 0);
//       camera_info.rot[1] = T(0, 1);
//       camera_info.rot[2] = T(0, 2);
//       camera_info.rot[3] = T(1, 0);
//       camera_info.rot[4] = T(1, 1);
//       camera_info.rot[5] = T(1, 2);
//       camera_info.rot[6] = T(2, 0);
//       camera_info.rot[7] = T(2, 1);
//       camera_info.rot[8] = T(2, 2);
//       camera_info.trans[0] = T(0, 3);
//       camera_info.trans[1] = T(1, 3);
//       camera_info.trans[2] = T(2, 3);
//       tex::TextureView texture_view(0, camera_info, blur_pano);
//       pano_views.emplace_back(texture_view);
//     }

//     tex::TexturePatches per_texture_patches;
//     GenerateMVETexturePatches(per_mve_mesh, per_texture_views, pano_views,
//                               per_texture_patches);

//     MergeTextureMesh(
//         per_texture_patches, building_texture_patches, per_mve_mesh,
//         building_texture_mesh,
//         Eigen::Vector3d(0, mesh_rooms_info_[index].camera_height, 0));
//   }

//   SaveBuildingTextureMesh(building_texture_mesh, building_texture_patches);
// }

void PanoRecons::MergeTextureMesh(const tex::TexturePatches &source_patches,
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

void PanoRecons::SaveBuildingTextureMesh(const mve::TriangleMesh::Ptr &mesh,
                                         tex::TexturePatches &texture_patches) {
  tex::TextureAtlases texture_atlases;
  tex::generate_texture_atlases(&texture_patches, settings_, &texture_atlases);

  AINFO << "Building obj model:" << options_.building_name << std::endl;
  tex::Model model;
  tex::build_model(mesh, texture_atlases, &model);

  if (!IsFolderExists(options_.output_dir)) {
    if (!createFolder(options_.output_dir)) {
      AWARN << "Can not creat path: " << options_.output_dir;
    }
  }

  std::string output_mesh = options_.output_dir + "/" + options_.building_name;
  tex::Model::save(model, output_mesh);
  AINFO << "Finished building obj model." << std::endl;
}

void PanoRecons::GenerateMVETexturePatches(
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

void PanoRecons::SimplifyMesh() {
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

std::vector<cv::Mat> PanoRecons::GenerateTextureImages(
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

void PanoRecons::Transform2WorldSpace(
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        &shot_cloud,
    const Eigen::Matrix4d &T) {
  for (int i = 0; i < shot_cloud.size(); ++i) {
    shot_cloud[i] = (T * shot_cloud[i].homogeneous()).hnormalized();
  }
}

// @todo
void PanoRecons::SetUserRoomHeight(const double &room_height) {
  options_.user_room_height = room_height;
}

}  // namespace lyj
