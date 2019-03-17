#include "ndpt.hpp"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

template <typename S, size_t N> void render(size_t nWorkers) {
  constexpr int dpi = 150;
  constexpr int hResPage = static_cast<int>(6.81102 * static_cast<double>(dpi));


  // Ratio of front cover is 173 : 246
  constexpr int vRes = static_cast<int>(
      (static_cast<double>(hResPage) / 173.) * 246.);

  constexpr int tmp = hResPage + static_cast<int>(0.551181 // 14 mm in inches
        * static_cast<double>(dpi));
  constexpr int hRes = hResPage + tmp;
  constexpr int hOffset = -tmp/2;

  std::cerr << "Rendering @" << dpi << " DPI (" << hRes << "x" << vRes << ")\n";

  Vec<S, N> origin{0};
  Vec<S, N> e0{1};
  Vec<S, N> e1{0, 1};
  Vec<S, N> e2{0, 0, 1};


  std::array<Vec<S, N>, N> floorAxes;
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      floorAxes[i][j] = (j == i + 1) ? 1 : 0;
  HyperCheckerboard<S, N> floor(Ray<S, N>(e0, (-e0).normalize()), floorAxes, 0);
  HyperCheckerboard<S, N> ceiling(Ray<S, N>(2.5*(-e0), e0.normalize()), floorAxes,0);
  Scene<S,N> scene{&floor, &ceiling};

  Camera<S, N> camera(Vec<S, N>{.0, -2., -2.}, // origin
                      Vec<S, N>{},          // centre
                      Vec<S, N>{-1.,.0,.0},        // down
                      Vec<S, N>{0, 1, -1},  // right
                      hRes, vRes);
  camera.centre = camera.origin * -.4;
  camera.hOffset = hOffset;

  for (size_t i = 0; i < (N - 3) / 2; i++) {
    camera.right[3 + 2 * i] = std::pow(2, -static_cast<S>(i) - 1);
    camera.down[4 + 2 * i] = std::pow(2, -static_cast<S>(i) - 1);
  }

  assert( std::abs(camera.right.dot(camera.down)) < eps<S> );
  assert( std::abs(camera.right.dot(camera.origin-camera.centre)) < eps<S> );
  assert( std::abs(camera.down.dot(camera.origin-camera.centre)) < eps<S> );


  camera.right = camera.right.normalize() * 0.9 / hResPage;
  camera.down = camera.down.normalize() * 0.9 / hResPage;

  PNGScreen<S> screen(camera.hRes, camera.vRes);
  Sampler<S, N, PNGScreen<S>> sampler(scene, camera, screen, 
      0.002, // target
      50 // minimal ray count
    );
  sampler.nWorkers = nWorkers;
  sampler.shoot();
  screen.png.write("sleeve.png");
}

int main(int argc, char *argv[]) {
  size_t nWorkers = 0;

  // Parse commandline options
  po::options_description desc("Allowed options");
  desc.add_options()("help", "this help message")(
      "threads", po::value<size_t>(), "number of worker threads to use");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << "\n";
    return 1;
  }

  if (vm.count("threads"))
    nWorkers = vm["threads"].as<size_t>();

  // Render!
  render<double, 12>(nWorkers);
  return 0;
}

// vim: sw=2 ts=2 et
