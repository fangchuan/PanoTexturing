#ifndef LYJ_SRC_BASE_PANORAMA_H_
#define LYJ_SRC_BASE_PANORAMA_H_

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "panorama.h"

namespace lyj {

class Panorama {
 public:
  Panorama() {}

  // 该构造函数会根据绝对路径自动解析出全景图的名字，例如：
  // absolute_path = "/Users/czh190502/Projects/datasets/a_for_modelling.jpg"
  // name_suffix = "_for_modelling"
  // 则解析出的名字 name_ = "a"
  //
  // 如果需要自定义名字（例如做特征点提取&匹配的评估时，不同文件夹下的图片会有同名的情况，则需要
  // 设定绝对路径为图片名字，作为唯一标识），则调用SetName()覆盖。
  Panorama(const cv::Mat &img, const std::string &absolute_path,
           const std::string &name_suffix = "");

  void Reset(const cv::Mat &img, const std::string &absolute_path,
             const std::string &name_suffix = "");

  inline void ClearMat();
  inline void LoadMat();

  inline cv::Mat GetImage() const;
  inline int Rows() const;
  inline int Cols() const;

  inline double GetCameraHeight() const;
  inline void SetCameraHeight(const double height);

  inline std::string GetName() const;
  inline void SetName(const std::string &name);
  // inline std::string GetSuffix() const;
  // inline std::string GetBaseName() const;
  inline std::string GetAbsolutePath() const;

  inline bool IsMainImage() const;
  inline void SetMainImage(const bool is_main);

  inline bool IsConnection() const;

  cv::Vec3b GetPixelColor(const int row, const int col) const;

  // 把全景图上的像素点投影到单位球上。
  Eigen::Vector3d UV2UnitBall(const Eigen::Vector2d &uv) const;

  // 把全景图上的像素点投影成相机坐标系下的三维点，depth是点到相机光心（相机坐标系原点）的距离。
  Eigen::Vector3d UV2XYZ(const Eigen::Vector2d &uv, const double depth) const;

  // 把相机坐标系下的三维点投影成全景图上的像素点。
  Eigen::Vector2d XYZ2UV(const Eigen::Vector3d &xyz) const;

  // 把全景图上的像素点投影到单位球上。
  // @input   uv     归一化的像素坐标
  // @return         单位球上的三维坐标
  static Eigen::Vector3d NormalizedUV2UnitBall(const Eigen::Vector2d &uv);

  // 把全景图上的像素点投影成相机坐标系下的三维点，depth是点到相机光心（相机坐标系原点）的距离。
  // @input   uv     归一化的像素坐标
  // @input   depth  深度
  // @return         三维坐标
  static Eigen::Vector3d NormalizedUV2XYZ(const Eigen::Vector2d &uv,
                                          const double depth);

  // 把相机坐标系下的三维点投影成全景图上的像素点。
  // @input   xyz   三维坐标
  // @return        归一化的像素坐标
  static Eigen::Vector2d XYZ2NormalizedUV(const Eigen::Vector3d &xyz);

  // 从全景图生成透视图
  // @param pano       全景图，列数=2*行数。
  // @param longitude  图片中心点视线的经度（longitude），取值范围[-pi, pi]。
  // @param latitude   图片中心点视线的维度（latitude），取值范围[-pi/2, pi/2]。
  // @param fov_v      图片竖直方向的 fov = atan(rows/2/r)，取值范围(0, pi)。
  // @param cols       生成图片的列数
  // @param rows       生成图片的行数
  static cv::Mat ImgLookAt(const cv::Mat &pano, double longitude,
                           double latitude, const double fov_v, const int cols,
                           const int rows);

 private:
  /* 为了兼容线上带后缀的命名方式，初始化Panorama时特地设置了"name_suffix"的入参，默认为空。

     例子1，name_suffix = ""
     absolute_path: "/Users/czh190502/Projects/datasets/a.jpg"
     basename: "a.jpg"
     name: "a"
     suffix: "jpg"

     例子2，name_suffix = "_for_modelling"
     absolute_path: "/Users/czh190502/Projects/datasets/a_for_modelling.jpg"
     basename: "a_for_modelling.jpg"
     name: "a"
     suffix: "jpg"
  */
  // 从图片路径解析出name_、suffix_、basename_等信息
  // @param  name_suffix  兼容线上带后缀的命名方式。
  void ParseImagePath(const std::string &name_suffix);

  std::string name_;  // 图片的唯一标识，Database中以name_来索引
  std::string absolute_path_;  // 图片的绝对路径

  cv::Mat img_;

  double camera_height_ = 1.6;  // 相机高度

  // 三维模型的结构和纹理都是根据主图片计算的，非主图片只会计算它相对于主图片的相对位置，以浏览
  // 点位的形式添加到模型中。增量式单空间建模打破了这个限制，没有主图片的概念，而是融合所有图片
  // 的信息。
  bool is_main_pano_ = true;

  // 是否是连接点位图片
  bool is_connection_ = false;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////
inline void Panorama::ClearMat() { img_.release(); }

inline void Panorama::LoadMat() {
  // img_ = cv::imread()
}

inline cv::Mat Panorama::GetImage() const { return img_; }
inline int Panorama::Rows() const { return img_.rows; }
inline int Panorama::Cols() const { return img_.cols; }

inline void Panorama::SetCameraHeight(const double h) { camera_height_ = h; }
inline double Panorama::GetCameraHeight() const { return camera_height_; }

inline std::string Panorama::GetName() const { return name_; }
inline std::string Panorama::GetAbsolutePath() const { return absolute_path_; }
inline void Panorama::SetName(const std::string &name) { name_ = name; }

inline bool Panorama::IsMainImage() const { return is_main_pano_; }
inline void Panorama::SetMainImage(const bool is_main) {
  is_main_pano_ = is_main;
}

inline bool Panorama::IsConnection() const { return is_connection_; }

}  // namespace lyj

#endif  // LYJ_SRC_BASE_PANORAMA_H_