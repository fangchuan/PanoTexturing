//
// Created by yuandong on 2022/2/27.
//

#ifndef LYJ_RECONSTRUCTION_DEV_SRC_MESH_UTILS_H_
#define LYJ_RECONSTRUCTION_DEV_SRC_MESH_UTILS_H_

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

// #include "base/database.h"
#include "cubemap.h"
// #include "floor_plan/base.h"
#include "open3d/geometry/TriangleMesh.h"
#include "tex/texturing.h"

// namespace open3d {
// namespace io {
//
// struct ReadTriangleMeshOptions {
//   /// Enables post-processing on the mesh
//   bool enable_post_processing = false;
//   /// Print progress to stdout about loading progress.
//   /// Also see \p update_progress if you want to have your own progress
//   /// indicators or to be able to cancel loading.
//   bool print_progress = false;
//   /// Callback to invoke as reading is progressing, parameter is percentage
//   /// completion (0.-100.) return true indicates to continue loading, false
//   /// means to try to stop loading and cleanup
//   std::function<bool(double)> update_progress;
// };
//
// bool ReadTriangleMeshFromOBJ(const std::string &filename,
//                              open3d::geometry::TriangleMesh &mesh,
//                              const ReadTriangleMeshOptions&);
//
// }
// }

// inline void SelectClosedShotsPair(const lyj::FloorPlan &room,
//                                   std::vector<int> &updated_index,
//                                   int &source_index, int &target_index) {
//   double min_dis = 1e5;
//   for (int i = 0; i < updated_index.size(); ++i) {
//     for (int j = 0; j < room.NumShots(); ++j) {
//       if (std::find(updated_index.begin(), updated_index.end(), j) !=
//           updated_index.end())
//         continue;
//       double dis = (room.GetShot(updated_index[i]).GetPosition() -
//                     room.GetShot(j).GetPosition())
//                        .norm();
//       if (dis < min_dis) {
//         min_dis = dis;
//         target_index = updated_index[i];
//         source_index = j;
//       }
//     }
//   }
//   updated_index.emplace_back(source_index);
// }

inline cv::Mat RotateImage(const cv::Mat &image, const float &rot_angle) {
  cv::Mat res_image;
  cv::Point center(image.cols / 2, image.rows / 2);
  cv::Mat rot_mat = cv::getRotationMatrix2D(center, rot_angle, 1.0);
  cv::warpAffine(image, res_image, rot_mat, image.size());
  return res_image;
}

inline cv::Mat FlipY(const cv::Mat &image) {
  cv::Mat res_image(image.size(), image.type());
  cv::flip(image, res_image, 1);
  return res_image;
}

inline cv::Mat FlipX(const cv::Mat &image) {
  cv::Mat res_image(image.size(), image.type());
  cv::flip(image, res_image, 0);
  return res_image;
}

inline Eigen::Matrix4f GetRotateX(const float &angle) {
  Eigen::Matrix4f res = Eigen::Matrix4f::Identity();
  res << 1.f, 0.f, 0.f, 0.f, 0.f, std::cos(angle), -std::sin(angle), 0.f, 0.f,
      std::sin(angle), std::cos(angle), 0.f, 0.f, 0.f, 0.f, 1.f;
  return res;
}

inline Eigen::Matrix4f GetRotateY(const float &angle) {
  Eigen::Matrix4f res = Eigen::Matrix4f::Identity();
  res << std::cos(angle), 0.f, std::sin(angle), 0.f, 0.f, 1.f, 0.f, 0.f,
      -std::sin(angle), 0.f, std::cos(angle), 0.f, 0.f, 0.f, 0.f, 1.f;
  return res;
}

inline void AddTextureViewsFromPano(const cv::Mat &pano,
                                    const bool &bottom_mask_flag,
                                    tex::TextureViews &texture_views) {
  // cv::Mat pano = database.GetPano(shot.GetName()).GetImage();

  double angle = M_PI / 2;
  // cut pano to cubemaps
  lyj::Cubemap cubemap;
  cubemap.GenerateFromPano(pano, 600, angle);

  std::vector<cv::Mat> texture_image;
  texture_image.resize(6);
  texture_image[2] = RotateImage(cubemap.Top(), -90);
  // cv::imwrite("top.jpg", texture_image[2]);
  texture_image[1] = RotateImage(cubemap.Bottom(), 90);
  // cv::imwrite("bottom.jpg", texture_image[1]);
  texture_image[0] = FlipX(FlipY(cubemap.Back()));
  // cv::imwrite("back.jpg", texture_image[0]);
  texture_image[3] = FlipX(FlipY(cubemap.Front()));
  // cv::imwrite("front.jpg", texture_image[3]);
  texture_image[5] = FlipX(FlipY(cubemap.Right()));
  // cv::imwrite("right.jpg", texture_image[5]);
  texture_image[4] = FlipX(FlipY(cubemap.Left()));
  // cv::imwrite("left.jpg", texture_image[4]);

  // 底部mask
  cv::Mat bottom_mask = cv::Mat::zeros(texture_image[1].size(), CV_8UC1);
  int mask_circle_radius = 80;
  cv::circle(bottom_mask,
             cv::Point(texture_image[1].cols / 2, texture_image[1].rows / 2),
             mask_circle_radius, 255, -1);
  //  cv::circle(texture_image[1],cv::Point(texture_image[1].cols/2,texture_image[1].rows/2),
  //  mask_circle_radius, 255, 2); cv::imshow("texture_image",texture_image[1]);
  //  cv::waitKey(0);

  // front bottom top back left right
  std::vector<Eigen::Matrix3d> cam_matrixs(6);
  cam_matrixs[0] << 1, 0, 0, 0, 1, 0, 0, 0, 1;    // front
  cam_matrixs[1] << 1, 0, 0, 0, 0, 1, 0, -1, 0;   // bottom
  cam_matrixs[2] << 1, 0, 0, 0, 0, -1, 0, 1, 0;   // top
  cam_matrixs[3] << -1, 0, 0, 0, 1, 0, 0, 0, -1;  // back
  cam_matrixs[4] << 0, 0, -1, 0, 1, 0, 1, 0, 0;   // left
  cam_matrixs[5] << 0, 0, 1, 0, 1, 0, -1, 0, 0;   // right

  // const auto T = shot.GetT();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  // T.block<3,3>(0,0) = Eigen::Quaterniond(0.7071067811865476,
  // 0.0,-0.7071067811865475,0.0).toRotationMatrix();
  T.block<3, 1>(0, 3) = Eigen::Vector3d(0, 1.6, 0);
  auto T_r = T.block<3, 3>(0, 0);
  Eigen::Vector3d T_t = Eigen::Vector3d(T(0, 3), T(1, 3), T(2, 3));

  Eigen::Matrix3d rotation = (T_r.transpose()).block<3, 3>(0, 0);
  Eigen::Vector3d translation =
      Eigen::Vector3d(-(T_t(0)), -(T_t(1)), -(T_t(2)));

  for (int i = 0; i < 6; ++i) {
    mve::CameraInfo camera_info;
    camera_info.flen = 0.5 / std::tan(angle / 2);
    auto update_matrix = (cam_matrixs[i]) * rotation;
    camera_info.rot[0] = update_matrix(0, 0);
    camera_info.rot[1] = update_matrix(0, 1);
    camera_info.rot[2] = update_matrix(0, 2);
    camera_info.rot[3] = update_matrix(1, 0);
    camera_info.rot[4] = update_matrix(1, 1);
    camera_info.rot[5] = update_matrix(1, 2);
    camera_info.rot[6] = update_matrix(2, 0);
    camera_info.rot[7] = update_matrix(2, 1);
    camera_info.rot[8] = update_matrix(2, 2);
    auto update_trans = update_matrix * translation;
    camera_info.trans[0] = update_trans(0);
    camera_info.trans[1] = update_trans(1);
    camera_info.trans[2] = update_trans(2);

    std::cout << " camera_" << i << ": \n";
    camera_info.debug_print();
    tex::TextureView texture_view(texture_views.size(), camera_info,
                                  texture_image[i]);
    if (i == 1 && bottom_mask_flag) {
      texture_view.AddValidMask(~bottom_mask);
    }

    texture_views.emplace_back(texture_view);
  }
}

inline Eigen::Vector2d XYZ2NormalizedUV(const Eigen::Vector3d &xyz) {
  double longitude = atan2(xyz[1], sqrt(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
  double latitude = atan2(xyz[0], -xyz[2]);
  double u = (latitude / (2 * M_PI) + 0.5);
  double v = (0.5 - longitude / M_PI);
  return Eigen::Vector2d(u, v);
}

inline Eigen::Vector2d XYZ2UV(const Eigen::Vector3d &xyz, const cv::Mat &equi) {
  Eigen::Vector2d uv_norm = XYZ2NormalizedUV(xyz);
  return Eigen::Vector2d(uv_norm[0] * equi.cols, uv_norm[1] * equi.rows);
}

inline cv::Vec3b GetPixelColor(const int row, const int col,
                               const cv::Mat &equi) {
  // img已经确保是cv::Vec3b类型
  if (row < 0 || row >= equi.rows || col < 0 || col >= equi.cols) {
    return cv::Vec3b(0, 0, 0);
  }
  return equi.at<cv::Vec3b>(row, col);
}

struct Line {
  Line() : x1(0), x2(0), y1(0), y2(0) {}  // 无参数的构造函数数组初始化时调用
  Line(const cv::Point2d &p1, const cv::Point2d &p2)
      : x1(p1.x), y1(p1.y), x2(p2.x), y2(p2.y) {}

  double x1;
  double y1;
  double x2;
  double y2;
};

static bool LineIntersection(const Line &l1, const Line &l2) {
  // 快速排斥实验
  if ((l1.x1 > l1.x2 ? l1.x1 : l1.x2) < (l2.x1 < l2.x2 ? l2.x1 : l2.x2) ||
      (l1.y1 > l1.y2 ? l1.y1 : l1.y2) < (l2.y1 < l2.y2 ? l2.y1 : l2.y2) ||
      (l2.x1 > l2.x2 ? l2.x1 : l2.x2) < (l1.x1 < l1.x2 ? l1.x1 : l1.x2) ||
      (l2.y1 > l2.y2 ? l2.y1 : l2.y2) < (l1.y1 < l1.y2 ? l1.y1 : l1.y2)) {
    return false;
  }
  // 跨立实验
  if ((((l1.x1 - l2.x1) * (l2.y2 - l2.y1) - (l1.y1 - l2.y1) * (l2.x2 - l2.x1)) *
       ((l1.x2 - l2.x1) * (l2.y2 - l2.y1) -
        (l1.y2 - l2.y1) * (l2.x2 - l2.x1))) > 0 ||
      (((l2.x1 - l1.x1) * (l1.y2 - l1.y1) - (l2.y1 - l1.y1) * (l1.x2 - l1.x1)) *
       ((l2.x2 - l1.x1) * (l1.y2 - l1.y1) -
        (l2.y2 - l1.y1) * (l1.x2 - l1.x1))) > 0) {
    return false;
  }
  return true;
}

// inline bool Visable(const lyj::FloorPlan &room, const lyj::Shot &shot,
//                     const Eigen::Vector3d &point) {
//   int cross_time = 0;
//   Line l1;
//   l1.x1 = shot.GetPosition()[0];
//   l1.y1 = shot.GetPosition()[2];
//   l1.x2 = point[0];
//   l1.y2 = point[2];
//   for (int i = 0; i < room.NumWalls(); ++i) {
//     Line l2;
//     l2.x1 = room.GetWall(i).GetRightEnd()[0] + 0.001;
//     l2.y1 = room.GetWall(i).GetRightEnd()[2] + 0.001;
//     l2.x2 = room.GetWall(i).GetLeftEnd()[0] + 0.001;
//     l2.y2 = room.GetWall(i).GetLeftEnd()[2] + 0.001;
//     if (LineIntersection(l1, l2)) {
//       cross_time++;
//     }
//   }
//   if (cross_time > 1) return false;
//   return true;
// }

bool hasOnlyAsciiCharacters(const std::string &string_to_test) {
  constexpr char kUpperAsciiBound = '~';
  constexpr char kLowerAsciiBound = ' ';
  for (const char &character : string_to_test) {
    if (character > kUpperAsciiBound || character < kLowerAsciiBound) {
      return false;
    }
  }
  return true;
}

bool IsFileExist(const std::string &file) {
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFREG)) {
    return true;
  }
  return false;
}

bool IsFolderExists(const std::string &path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

bool createFolder(const std::string &path_to_create_input) {
  constexpr mode_t kMode = 0777;

  CHECK(!path_to_create_input.empty()) << "Cannot create empty path!";

  // Append slash if necessary to make sure that stepping through the folders
  // works.
  std::string path_to_create = path_to_create_input;
  if (path_to_create.back() != '/') {
    path_to_create += '/';
  }

  // Loop over the path and create one folder after another.
  size_t current_position = 0u;
  size_t previous_position = 0u;
  std::string current_directory;
  while ((current_position = path_to_create.find_first_of(
              '/', previous_position)) != std::string::npos) {
    current_directory = path_to_create.substr(0, current_position++);
    previous_position = current_position;

    if (current_directory == "." || current_directory.empty()) {
      continue;
    }

    if (!hasOnlyAsciiCharacters(current_directory)) {
      return false;
    }

    int make_dir_status = 0;
    if ((make_dir_status = mkdir(current_directory.c_str(), kMode)) &&
        errno != EEXIST) {
      VLOG(2) << "Unable to make path! Error: " << strerror(errno);
      return make_dir_status == 0;
    }
  }
  return true;
}

#endif  // LYJ_RECONSTRUCTION_DEV_SRC_MESH_UTILS_H_
