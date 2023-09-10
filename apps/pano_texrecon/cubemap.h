#ifndef LYJ_SRC_BASE_CUBEMAP_H_
#define LYJ_SRC_BASE_CUBEMAP_H_

#include <math.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "panorama.h"
// #include "util/error_type.h"

namespace lyj {

class Cubemap {
 public:
  enum Face {
    kLeft = 0,
    kFront = 1,
    kRight = 2,
    kBack = 3,
    kTop = 4,
    kBottom = 5
  };

  inline cv::Mat Left() const;
  inline cv::Mat Front() const;
  inline cv::Mat Right() const;
  inline cv::Mat Back() const;
  inline cv::Mat Top() const;
  inline cv::Mat Bottom() const;

  cv::Mat GetFace(int index) const;

  // 从全景图生成cubemap，全景图满足 2*行数 = 列数。
  // @param resolution 生成的cubemap六个面的分辨率。
  //                   如果不包含这个参数，则分辨率是自适应得出的，
  //                   使得 6*x^2 = y.cols*y.rows。
  bool GenerateFromPano(const cv::Mat &pano, const int resolution, double fov);
  bool GenerateFromPano(const lyj::Panorama &pano, int resolution);
  bool GenerateFromPano(const cv::Mat &pano, int resolution);
  bool GenerateFromPano(const lyj::Panorama &pano);
  bool GenerateFromPano(const cv::Mat &pano);

  // 把cubemap某个面上的点转换到全景图上。
  static Eigen::Vector2d Transfer2Pano(const Eigen::Vector2d pt,
                                       const int pano_rows, const int pano_cols,
                                       const int resolution,
                                       const Cubemap::Face face);
  Eigen::Vector2d Transfer2Pano(const Eigen::Vector2d pt,
                                const Cubemap::Face face) const;

  void Save(std::string dir, std::string prefix = "");

 private:
  // cubemap的分辨率
  int resolution_;

  // 生成当前cubemap的全景图的分辨率
  int pano_rows_, pano_cols_;

  cv::Mat left_, front_, right_, back_, top_, bottom_;

  // cubemap六个面的中点的经度longitude（对应panorama的列数）
  static const std::vector<double> center_longs_;
  // cubemap六个面的中点的纬度latitude（对应panorama的行数）
  static const std::vector<double> center_lats_;
};

/******************************************************************
  Description: 切图接口，线上用于cubemap切图
  Input:  input_dri: 校正好的原图文件路径，文件夹下包含校正好的原图；
          prefix: 文件名；
          format: 文件类型；
          output_dir: 结果输出路径，包括6张2K*2K分辨率的切片图；
  Output: msg: 错误信息。
******************************************************************/
int CutPanoToCubic(std::string input_dir, std::string prefix,
                   std::string format, std::string output_dir);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

inline cv::Mat Cubemap::Left() const { return left_; }
inline cv::Mat Cubemap::Front() const { return front_; }
inline cv::Mat Cubemap::Right() const { return right_; }
inline cv::Mat Cubemap::Back() const { return back_; }
inline cv::Mat Cubemap::Top() const { return top_; }
inline cv::Mat Cubemap::Bottom() const { return bottom_; }

}  // namespace lyj

#endif  // LYJ_SRC_BASE_CUBEMAP_H_