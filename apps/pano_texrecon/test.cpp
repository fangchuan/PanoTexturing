#include <iostream>
#include <fstream>
#include <vector>
#include <tbb/task_scheduler_init.h>
#include <omp.h>

#include <util/timer.h>
#include <util/system.h>
#include <util/file_system.h>
#include <mve/mesh_io_ply.h>

#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"

#include "pano_recons.h"

using namespace lyj;

int main(int argc, char** argv) {
  util::system::print_build_timestamp(argv[0]);
  util::system::register_segfault_handler();

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " INPUT_PANO_FILEPATH INPUT_O3D_MESH_FILEPATH OUTPUT_DIR"
              << std::endl;
    return -1;
  }
  Timer timer;
  util::WallTimer wtimer;

  const std::string input_pano_filepath = argv[1];
  const std::string input_o3d_mesh_filepath = argv[2];
  const std::string output_dir = argv[3];

  lyj::PanoReconsOptions options;
  options.output_dir = output_dir;
  options.obj_filepath = input_o3d_mesh_filepath;
  options.simplify_mesh = false;
  options.whole_building_unseen_fill = true;
  options.simplify_voxel_size = 0.1;
  options.remove_tiny_mesh = false;

  PanoRecons pano_recons(options, input_pano_filepath);
  pano_recons.Build();
    // tex::Model model;
    // // tex::build_model(mesh, texture_atlases, &model);
    // std::cout << "\tSaving model... " << std::flush;
    // tex::Model::save(model, "_view_selection");
    // std::cout << "done." << std::endl;
  return 0;
}