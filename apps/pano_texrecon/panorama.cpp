#include "panorama.h"

#include <math.h>

#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "log.h"

namespace lyj {

Panorama::Panorama(const cv::Mat &img, const std::string &absolute_path,
                   const std::string &name_suffix)
    : img_(img), absolute_path_(absolute_path) {
  ParseImagePath(name_suffix);
}

void Panorama::Reset(const cv::Mat &img, const std::string &absolute_path,
                     const std::string &name_suffix) {
  img_ = img;
  absolute_path_ = absolute_path;
  ParseImagePath(name_suffix);
}

void Panorama::ParseImagePath(const std::string &name_suffix) {
  std::string basename = absolute_path_.substr(absolute_path_.rfind('/') + 1);
  name_ = basename.substr(0, basename.rfind('.'));

  if (!name_suffix.empty()) {
    int length = name_.length() - name_suffix.length();
    if (length <= 0 || name_.substr(length) != name_suffix) {
      return;
    }
    name_ = basename.substr(0, basename.rfind('.') - name_suffix.length());
  }
}

cv::Vec3b Panorama::GetPixelColor(const int row, const int col) const {
  // img已经确保是cv::Vec3b类型
  if (row < 0 || row >= img_.rows || col < 0 || col >= img_.cols) {
    return cv::Vec3b(0, 0, 0);
  }
  return img_.at<cv::Vec3b>(row, col);
}

Eigen::Vector3d Panorama::UV2UnitBall(const Eigen::Vector2d &uv) const {
  Eigen::Vector2d uv_norm =
      Eigen::Vector2d(uv[0] / img_.cols, uv[1] / img_.rows);
  return NormalizedUV2UnitBall(uv_norm);
}

Eigen::Vector3d Panorama::UV2XYZ(const Eigen::Vector2d &uv,
                                 const double depth) const {
  return UV2UnitBall(uv) * depth;
}

Eigen::Vector2d Panorama::XYZ2UV(const Eigen::Vector3d &xyz) const {
  Eigen::Vector2d uv_norm = XYZ2NormalizedUV(xyz);
  return Eigen::Vector2d(uv_norm[0] * img_.cols, uv_norm[1] * img_.rows);
}

Eigen::Vector3d Panorama::NormalizedUV2UnitBall(const Eigen::Vector2d &uv) {
  //  pi/2  --------
  //       |        |
  //       |        |
  // -pi/2  --------
  //      -pi       pi
  double latitude = (uv[0] - 0.5) * 2.0 * M_PI;
  double longitude = (0.5 - uv[1]) * M_PI;
  double x = cos(longitude) * sin(latitude);
  double y = sin(longitude);
  double z = -cos(longitude) * cos(latitude);
  return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d Panorama::NormalizedUV2XYZ(const Eigen::Vector2d &uv,
                                           const double depth) {
  return NormalizedUV2UnitBall(uv) * depth;
}

Eigen::Vector2d Panorama::XYZ2NormalizedUV(const Eigen::Vector3d &xyz) {
  double longitude = atan2(xyz[1], sqrt(xyz[0] * xyz[0] + xyz[2] * xyz[2]));
  double latitude = atan2(xyz[0], -xyz[2]);
  double u = (latitude / (2 * M_PI) + 0.5);
  double v = (0.5 - longitude / M_PI);
  return Eigen::Vector2d(u, v);
}

cv::Mat Panorama::ImgLookAt(const cv::Mat &pano, double center_long,
                            double center_lat, const double fov_v,
                            const int cols, const int rows) {
  assert(pano.type() == 16);  // CV_8UC3
  assert(center_long >= -M_PI && center_long <= M_PI);
  assert(center_lat >= -M_PI / 2 && center_lat <= M_PI / 2);
  assert(fov_v > 0 && fov_v < M_PI);

  cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);

  double r = rows / 2.0 / tan(fov_v / 2.0);
  double r_2 = r * r;
  double cols_half = (cols - 1) / 2.0;
  double rows_half = (rows - 1) / 2.0;

  int pixels = cols * rows;

  // #ifdef OPENMP_ENABLED

  //   // omp speedup
  //   int num_threads_ = omp_get_num_procs();

  // #pragma omp parallel num_threads(num_threads_)
  //   {
  // #pragma omp for nowait
  // #endif

  // 按照像素数量使用指针方式和remap函数
  if (pixels > 1e5) {
    cv::Mat map_x, map_y;
    map_x.create(image.size(), CV_32FC1);
    map_y.create(image.size(), CV_32FC1);
    float *data_mapx = map_x.ptr<float>(0);
    float *data_mapy = map_y.ptr<float>(0);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        double du = j - cols_half;
        double dv = i - rows_half;
        double R = sqrt(dv * dv + r_2);
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

        double u_raw = (longitude / (2 * M_PI) + 0.5) * pano.cols;
        double v_raw = (0.5 - latitude / M_PI) * pano.rows;
        *(data_mapx + i * cols + j) = static_cast<float>(u_raw);
        *(data_mapy + i * cols + j) = static_cast<float>(v_raw);
      }
    }
    cv::remap(pano, image, map_x, map_y, cv::INTER_LANCZOS4, cv::BORDER_WRAP);
  } else {
    cv::Mat pano_clone = pano.clone();
    for (size_t i = 0; i < rows; i++) {
      uchar *data = image.ptr<uchar>(i);

      for (size_t j = 0; j < cols; j++) {
        double du = j - cols_half;
        double dv = i - rows_half;
        double R = sqrt(dv * dv + r_2);
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

        double u_raw = (longitude / (2 * M_PI) + 0.5) * pano.cols;
        double v_raw = (0.5 - latitude / M_PI) * pano.rows;
        if (u_raw < 0) {
          u_raw += pano.cols - 1;
        }
        if (u_raw > pano.cols - 1) {
          u_raw -= pano.cols - 1;
        }
        if (v_raw < 0) {
          v_raw += pano.rows - 1;
        }
        if (v_raw > pano.rows - 1) {
          v_raw -= pano.rows - 1;
        }
        int Px1 = floor(u_raw);
        int Px2 = ceil(u_raw);
        int Py1 = floor(v_raw);
        int Py2 = ceil(v_raw);
        double r = 0, g = 0, b = 0;
        for (int k_row = Py1; k_row <= Py2; k_row++) {
          uchar *dataImg = pano_clone.ptr<uchar>(k_row);

          for (int k_col = Px1; k_col <= Px2; k_col++) {
            double weight =
                (1 - std::fabs(u_raw - k_col)) * (1 - std::fabs(v_raw - k_row));

            b += weight * dataImg[3 * k_col];
            g += weight * dataImg[3 * k_col + 1];
            r += weight * dataImg[3 * k_col + 2];
          }
        }

        data[3 * j + 0] = int(b);
        data[3 * j + 1] = int(g);
        data[3 * j + 2] = int(r);
      }
    }
  }
  // #ifdef OPENMP_ENABLED
  //   }
  // #endif

  if (image.empty()) {
    std::cout << "empty image" << std::endl;
  }

  return image;
}
/*
double LanczosKernel(double x, int filter_size = 5) {
  if (x == 0) { return 1; }

  const double xp = M_PI * x;
  return filter_size * std::sin(xp) * std::sin(xp / 5) / (xp * xp);
}

cv::Vec3b GetWeightedPixelColor(const cv::Mat &pano,
                                double u,
                                double v,
                                int half_window = 5) {

  int window = 2 * half_window;
  const int row_max = pano.rows - 1;
  const int col_max = pano.cols - 1;
  std::vector<double> kernel_x, kernel_y;
  kernel_x.resize(window);
  kernel_y.resize(window);
  for (size_t i = 0; i < window; i++) {
    //kernel_x[i] = LanczosKernel()
  }
}

cv::Mat Panorama::ImgLookAtNew(const cv::Mat &pano, double center_long,
                               double center_lat, const double fov_v,
                               const int cols, const int rows) {
  assert(pano.type() == 16);  // CV_8UC3
  assert(center_long >= -M_PI && center_long <= M_PI);
  assert(center_lat >= -M_PI / 2 && center_lat <= M_PI / 2);
  assert(fov_v > 0 && fov_v < M_PI);

  cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);

  double r = rows / 2.0 / tan(fov_v / 2.0);
  double r_2 = r * r;
  double cols_half = (cols - 1) / 2.0;
  double rows_half = (rows - 1) / 2.0;

  int filter_size = 5;

  cv::Mat pano_clone = pano.clone();
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      double du = j - cols_half;
      double dv = i - rows_half;
      double R = sqrt(dv * dv + r_2);
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

      double u_raw = (longitude / (2 * M_PI) + 0.5) * pano.cols;
      double v_raw = (0.5 - latitude / M_PI) * pano.rows;
      if (u_raw < 0) {
        u_raw += pano.cols - 1;
      }
      if (u_raw > pano.cols - 1) {
        u_raw -= pano.cols - 1;
      }
      if (v_raw < 0) {
        v_raw += pano.rows - 1;
      }
      if (v_raw > pano.rows - 1) {
        v_raw -= pano.rows - 1;
      }

      int Px1 = floor(u_raw);
      int Px2 = ceil(u_raw);
      int Py1 = floor(v_raw);
      int Py2 = ceil(v_raw);
      double r = 0, g = 0, b = 0;
      for (int k_row = Py1; k_row <= Py2; k_row++) {
        uchar *dataImg = pano_clone.ptr<uchar>(k_row);

        for (int k_col = Px1; k_col <= Px2; k_col++) {
          double weight =
              (1 - std::fabs(u_raw - k_col)) * (1 - std::fabs(v_raw - k_row));

          b += weight * dataImg[3 * k_col];
          g += weight * dataImg[3 * k_col + 1];
          r += weight * dataImg[3 * k_col + 2];
        }
      }
    }
  }

  if (image.empty()) {
    std::cout << "empty image" << std::endl;
  }

  return image;
}
*/
}  // namespace lyj
