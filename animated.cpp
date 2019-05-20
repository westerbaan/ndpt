#include "ndpt.hpp"

#include <boost/program_options.hpp>

#include <sstream>

namespace po = boost::program_options;

template <typename S, size_t N> void render(size_t nWorkers, int frame, int nFrames) {
  constexpr int dpi = 75;
  constexpr int hResPage = static_cast<int>(6.81102 * static_cast<double>(dpi));

  // Ratio of front cover is 173 : 246
  constexpr int vRes = static_cast<int>(
      (static_cast<double>(hResPage) / 173.) * 246.);

  constexpr int hRes = hResPage;
  constexpr int hOffset = 0;

  std::cerr
    << "Rendering frame " << frame << "/" << nFrames
    << "  @" << dpi << " DPI (" << hRes << "x" << vRes << ")\n";

  Vec<S, N> origin{0};
  Vec<S, N> e0{1};
  Vec<S, N> e1{0, 1};
  Vec<S, N> e2{0, 0, 1};

  // ReflectiveSphere<S,N> sphere(origin, .9);

  // ReflectiveBowl<S,N> sphere(origin, .9, -e0);
  // ReflectiveBowl<S,N> sphere2(origin + 1.5*e1+1.5*e2 + .3*e0, .9, e0);

  auto shade = blue<S>*.5;
  S frameRatio = S(frame) / S(nFrames-1);
  // ReflectiveBowl<S,N> sphere(origin - .2*(e1-e2), .9, e1-e2 - .5*e2, shade);
  // ReflectiveBowl<S,N> sphere2(origin + .2*(e1-e2), .9, -(e1-e2) - .5*e2, shade);
  ReflectiveBowl<S,N> sphere(origin - .2*frameRatio*(e1-e2), .9, e1-e2 - frameRatio*.5*e2, shade);
  ReflectiveBowl<S,N> sphere2(origin + .2*frameRatio*(e1-e2), .9, -(e1-e2) - frameRatio*.5*e2, shade);

  //ReflectiveTorus<S, N> torus(origin, .636);
  std::array<Vec<S, N>, N> floorAxes;
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      floorAxes[i][j] = (j == i + 1) ? 1 : 0;
  HyperCheckerboard<S, N> floor(Ray<S, N>(e0, (-e0).normalize()), floorAxes);
  HyperCheckerboard<S, N> ceiling(Ray<S, N>(2.5*(-e0), e0.normalize()), floorAxes);
  Scene<S,N> scene{&sphere, &sphere2, &floor, &ceiling};
  //Scene<S, N> scene{&torus, &floor, &ceiling};

  Camera<S, N> camera(Vec<S, N>{-2., -2., -2.}, // origin
                      Vec<S, N>{},          // centre
                      // Vec<S, N>{-1.,.5,0.5},        // down
                      Vec<S, N>{1.,-.5,-0.5},        // down
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
    // 0.002, // target
    // 50 // minimal ray count
    0.01,
    10
  );
  sampler.nWorkers = nWorkers;
  sampler.shoot();

  std::ostringstream filename;
  filename << "animated." << frame << ".png";
  screen.png.write(filename.str());
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
  auto nFrames = 100;
  for (int frame = 0; frame < nFrames; frame++) {
    render<double, 12>(nWorkers, frame, nFrames);
  }
  // render<double, 3>(nWorkers);
  return 0;
}

// vim: sw=2 ts=2 et
