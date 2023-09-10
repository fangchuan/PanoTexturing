#include "cubemap.h"

#include <iostream>
#include <opencv2/highgui.hpp>

#include "log.h"
#include "util/timer.h"

namespace lyj {

const std::vector<double> Cubemap::center_longs_ = {M_PI_2, 0,       -M_PI_2,
                                                    M_PI,   -M_PI_2, -M_PI_2};
const std::vector<double> Cubemap::center_lats_ = {0, 0, 0, 0, M_PI_2, -M_PI_2};

/*
const std::vector<double> Cubemap::center_longs_ = {M_PI_2, 0,       -M_PI_2,
                                                    M_PI,   -M_PI_2, -M_PI_2};
const std::vector<double> Cubemap::center_lats_ = {0, 0, 0, 0, M_PI_2, -M_PI_2};
*/

cv::Mat Cubemap::GetFace(int index) const {
  if (index == Face::kLeft) {
    return left_;
  } else if (index == Face::kFront) {
    return front_;
  } else if (index == Face::kRight) {
    return right_;
  } else if (index == Face::kBack) {
    return back_;
  } else if (index == Face::kTop) {
    return top_;
  } else if (index == Face::kBottom) {
    return bottom_;
  }
  cv::Mat empty;
  return empty;
}

bool Cubemap::GenerateFromPano(const cv::Mat &pano, const int resolution,
                               double fov) {
  if (pano.empty()) {
    return false;
  }
  pano_rows_ = pano.rows;
  pano_cols_ = pano.cols;
  resolution_ = resolution;
  left_ = Panorama::ImgLookAt(pano, center_longs_[0], center_lats_[0], fov,
                              resolution_, resolution_);
  front_ = Panorama::ImgLookAt(pano, center_longs_[1], center_lats_[1], fov,
                               resolution_, resolution_);
  right_ = Panorama::ImgLookAt(pano, center_longs_[2], center_lats_[2], fov,
                               resolution_, resolution_);
  back_ = Panorama::ImgLookAt(pano, center_longs_[3], center_lats_[3], fov,
                              resolution_, resolution_);
  top_ = Panorama::ImgLookAt(pano, center_longs_[4], center_lats_[4], fov,
                             resolution_, resolution_);
  bottom_ = Panorama::ImgLookAt(pano, center_longs_[5], center_lats_[5], fov,
                                resolution_, resolution_);
  return true;
}

bool Cubemap::GenerateFromPano(const cv::Mat &pano) {
  if (pano.empty()) {
    return false;
  }

  return GenerateFromPano(
      pano, static_cast<int>(std::sqrt(pano.rows * pano.cols / 6.0)));
}

bool Cubemap::GenerateFromPano(const cv::Mat &pano, int resolution) {
  if (pano.empty()) {
    return false;
  }
  pano_rows_ = pano.rows;
  pano_cols_ = pano.cols;
  resolution_ = resolution;
  left_ = Panorama::ImgLookAt(pano, center_longs_[0], center_lats_[0], M_PI_2,
                              resolution_, resolution_);
  front_ = Panorama::ImgLookAt(pano, center_longs_[1], center_lats_[1], M_PI_2,
                               resolution_, resolution_);
  right_ = Panorama::ImgLookAt(pano, center_longs_[2], center_lats_[2], M_PI_2,
                               resolution_, resolution_);
  back_ = Panorama::ImgLookAt(pano, center_longs_[3], center_lats_[3], M_PI_2,
                              resolution_, resolution_);
  top_ = Panorama::ImgLookAt(pano, center_longs_[4], center_lats_[4], M_PI_2,
                             resolution_, resolution_);
  bottom_ = Panorama::ImgLookAt(pano, center_longs_[5], center_lats_[5], M_PI_2,
                                resolution_, resolution_);
  return true;
}

bool Cubemap::GenerateFromPano(const lyj::Panorama &pano, int resolution) {
  return GenerateFromPano(pano.GetImage(), resolution);
}

bool Cubemap::GenerateFromPano(const lyj::Panorama &pano) {
  return GenerateFromPano(pano.GetImage());
}

Eigen::Vector2d Cubemap::Transfer2Pano(const Eigen::Vector2d pt,
                                       const Cubemap::Face face) const {
  return Transfer2Pano(pt, pano_rows_, pano_cols_, resolution_, face);
}

Eigen::Vector2d Cubemap::Transfer2Pano(const Eigen::Vector2d pt,
                                       const int pano_rows, const int pano_cols,
                                       const int resolution,
                                       const Cubemap::Face face) {
  double center_long = center_longs_.at(face);
  double center_lat = center_lats_.at(face);

  double du = pt[0] - (resolution - 1) / 2.0;
  double dv = pt[1] - (resolution - 1) / 2.0;
  double r = resolution / 2.0 / tan(M_PI_4);
  double R = sqrt(dv * dv + r * r);
  double dlat_c = atan(-dv / r);
  double latitude_c = center_lat + dlat_c;
  double h = R * sin(latitude_c);
  double s = R * cos(latitude_c);
  double latitude = atan(h / sqrt(s * s + du * du));
  double dlong = atan(du / s);
  double longitude = center_long + dlong;
  if (fabs(latitude_c) > M_PI_2) {
    longitude += M_PI;
  }
  if (longitude > M_PI) {
    longitude -= 2 * M_PI;
  } else if (longitude < -M_PI) {
    longitude += 2 * M_PI;
  }
  double u_pano = (longitude + M_PI) / 2.0 / M_PI * pano_cols;
  double v_pano = (M_PI_2 - latitude) / M_PI * pano_rows;

  return Eigen::Vector2d(u_pano, v_pano);
}

void Cubemap::Save(std::string dir, std::string prefix) {
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_left.jpg",
      left_);
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_front.jpg",
      front_);
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_right.jpg",
      right_);
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_back.jpg",
      back_);
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_top.jpg", top_);
  cv::imwrite(
      dir + "/" + prefix + "_for_modelling.jpg_perspective_view_bottom.jpg",
      bottom_);
}

int CutPanoToCubic(std::string input_dir, std::string prefix,
                   std::string format, std::string output_dir) {
  std::string pano_path = input_dir + "/" + prefix + "." + format;
  cv::Mat ori_Img = cv::imread(pano_path);

  if (ori_Img.empty()) {
    // msg.success = false;
    // msg.code = -1;
    // msg.message = "failed to read image, please check the path";
    AWARN << "failed to read image, please check the path";

    return -1;
  }

  // lyj::Timer timer;
  // timer.Start();
  util::WallTimer timer;
  lyj::Cubemap cube;
  int res = 2000;
  int rt = cube.GenerateFromPano(ori_Img, res);

  cube.Save(output_dir, prefix);
  // timer.End("---------- Cut pano to Cubic cost: ");
  AINFO << "Whole texturing procedure took: " << timer.get_elapsed_sec()
        << "s\n";

  // msg.success = true;
  // msg.code = 1;
  // msg.message = "Successfully cut pano to cubic!";
  AINFO << "Successfully cut pano to cubic!\n";

  return 1;
}

}  // namespace lyj
