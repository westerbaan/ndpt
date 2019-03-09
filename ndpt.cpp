#include <cassert>
#include <cmath>

#include <condition_variable>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>
#include <png++/png.hpp>

namespace po = boost::program_options;

template <typename S> constexpr S eps = std::numeric_limits<S>::epsilon();

template <typename S, size_t N> class UVec; // unit vector, see below

template <typename S, size_t N> class Body;

template <typename S, size_t N> class Interaction;

template <typename S> class Colour {
public:
  S R, G, B;

  constexpr Colour(const S R, const S G, const S B) : R(R), G(G), B(B) {}
  Colour() : R(0), G(0), B(0) {}

  // Adds two colours
  inline Colour<S> &operator+=(const Colour<S> &rhs) {
    this->R += rhs.R;
    this->G += rhs.G;
    this->B += rhs.B;
    return *this;
  }
  inline const Colour<S> operator+(const Colour<S> &rhs) const {
    return Colour<S>(*this) += rhs;
  }

  // Substracts colours
  inline Colour<S> &operator-=(const Colour<S> &rhs) {
    this->R -= rhs.R;
    this->G -= rhs.G;
    this->B -= rhs.B;
    return *this;
  }
  inline const Colour<S> operator-(const Colour<S> &rhs) const {
    return Colour<S>(*this) -= rhs;
  }

  // Scalar multiplication
  inline const Colour<S> operator*=(const S &s) {
    this->R *= s;
    this->G *= s;
    this->B *= s;
    return *this;
  }
  inline const Colour<S> operator*(const S &rhs) const {
    return Colour<S>(*this) *= rhs;
  }
  inline friend const Colour<S> operator*(const S &lhs, const Colour<S> &rhs) {
    return Colour<S>(rhs) *= lhs;
  }
  inline const Colour<S> operator/(const S &rhs) const {
    return *this * (1 / rhs);
  }

  // Supremum norm
  inline S supNorm() const {
    return std::max(this->R, std::max(this->G, this->B));
  }

  // Implicit casts
  inline operator png::rgb_pixel() const {
    return png::rgb_pixel(R * 255, G * 255, B * 255);
  }

  // Printing Colour
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const Colour<S> &col) {
    return os << '(' << col.R << ", " << col.G << ", " << col.B << ')';
  }
};

template <typename S> constexpr Colour<S> black = Colour<S>(0, 0, 0);

template <typename S> constexpr Colour<S> white = Colour<S>(1, 1, 1);

template <typename S> constexpr Colour<S> red = Colour<S>(1, 0, 0);

// Represents a vector of N-elements with scalar of type S
template <typename S, size_t N> class Vec {
protected:
  // Components of the vector
  std::array<S, N> vals;

public:
  // Uninitialized vector
  Vec<S, N>() {}

  // Copy constructor
  Vec<S, N>(const Vec<S, N> &other) = default;

  // Initializer list constructor
  template <typename T> Vec<S, N>(const std::initializer_list<T> &l) {
    // For some reason clang doesn't think l.size() is constexpr
    // static_assert(l.size() < N, "initializer_list too long");
    assert(l.size() < N);
    std::copy(l.begin(), l.end(), this->vals.begin());
    std::fill(&this->vals[l.size()], this->vals.end(), 0);
  }

  // Adds the given vector to this one.
  inline Vec<S, N> &operator+=(const Vec<S, N> &rhs) {
    for (size_t i = 0; i < N; i++)
      this->vals[i] += rhs.vals[i];
    return *this;
  }

  // Adds two vectors
  inline const Vec<S, N> operator+(const Vec<S, N> &rhs) const {
    return Vec<S, N>(*this) += rhs;
  }

  // Substracts the given vector from this one.
  inline Vec<S, N> &operator-=(const Vec<S, N> &rhs) {
    for (size_t i = 0; i < N; i++)
      this->vals[i] -= rhs.vals[i];
    return *this;
  }

  // Substracts two vectors
  inline const Vec<S, N> operator-(const Vec<S, N> &rhs) const {
    return Vec<S, N>(*this) -= rhs;
  }

  // Inverts vector
  inline const Vec<S, N> operator-() const {
    Vec<S, N> ret;
    for (size_t i = 0; i < N; i++)
      ret.vals[i] = -this->vals[i];
    return ret;
  }

  // Scalar multiplication in-place
  inline const Vec<S, N> operator*=(const S &s) {
    for (size_t i = 0; i < N; i++)
      this->vals[i] *= s;
    return *this;
  }

  // Scalar multiplication on the right
  inline const Vec<S, N> operator*(const S &rhs) const {
    return Vec<S, N>(*this) *= rhs;
  }

  // Scalar multiplication on the left
  inline friend const Vec<S, N> operator*(const S &lhs, const Vec<S, N> &rhs) {
    return Vec<S, N>(rhs) *= lhs;
  }

  // Division scalar multiplication
  inline const Vec<S, N> operator/(const S &rhs) const {
    return *this * (1 / rhs);
  }

  // Printing vectors
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const Vec<S, N> &vec) {
    os << '[';
    if (N != 0)
      os << vec.vals[0];
    for (size_t i = 1; i < N; i++)
      os << ' ' << vec.vals[i];
    os << ']';
    return os;
  }

  // Pointwise product of vectors
  inline friend Vec<S, N> pwProd(const Vec<S, N> &v, const Vec<S, N> &w) {
    Vec<S, N> ret;
    for (size_t i = 0; i < N; i++)
      ret.vals[i] = v.vals[i] * w.vals[i];
    return ret;
  }

  // Compute inner product
  inline S dot(const Vec<S, N> &other) const {
    S ret = 0;
    for (int i = 0; i < N; i++)
      ret += this->vals[i] * other.vals[i];
    return ret;
  }

  // Length of vector
  inline S length() const { return std::sqrt(this->dot(*this)); }

  // Subscript operator
  inline S &operator[](const size_t i) { return vals[i]; }

  const UVec<S, N> normalize() const;
};

// Represents a unit-vector vector of N-elements with scalar of type S
template <typename S, size_t N> class UVec {
  Vec<S, N> v;
  UVec<S, N>() {}

public:
  // Create a unit vector by normalizing a regular vector
  inline static const UVec<S, N> normalize(const Vec<S, N> &v) {
    UVec<S, N> ret;
    ret.v = v / v.length();
    return ret;
  }

  // Reflect this vector over the given normal
  inline UVec<S, N> reflect(const UVec<S, N> &normal) const {
    return (*this - normal * (2 * this->dot(normal))).normalize();
  }

  // Cast to vector
  inline operator Vec<S, N>() const { return this->v; }

  // Implicit casts and templates don't combine well, so we have to add
  // quite some boilerplate to make the compiler understand uv1 + uv2
  // actually means (vec)uv1 + (vec)uv2.
  inline S dot(const Vec<S, N> &other) const {
    return static_cast<Vec<S, N>>(*this).dot(other);
  }
  inline const Vec<S, N> operator*(const S &rhs) const {
    return Vec<S, N>(*this) *= rhs;
  }
  inline const Vec<S, N> operator/(const S &rhs) const {
    return static_cast<Vec<S, N>>(*this) / rhs;
  }
  inline const Vec<S, N> operator-(const Vec<S, N> &rhs) const {
    return static_cast<Vec<S, N>>(*this) - rhs;
  }
  inline const Vec<S, N> operator-(const UVec<S, N> &rhs) const {
    return *this - static_cast<Vec<S, N>>(rhs);
  }
  inline const Vec<S, N> operator-() const {
    return -static_cast<Vec<S, N>>(*this);
  }
  inline const Vec<S, N> operator+(const Vec<S, N> &rhs) const {
    return static_cast<Vec<S, N>>(*this) + rhs;
  }
  inline const Vec<S, N> operator+(const UVec<S, N> &rhs) const {
    return *this + static_cast<Vec<S, N>>(rhs);
  }
  inline friend std::ostream &operator<<(std::ostream &os,
                                         const UVec<S, N> &v) {
    return os << static_cast<Vec<S, N>>(v);
  }
};

// Normalize vector
template <typename S, size_t N>
inline const UVec<S, N> Vec<S, N>::normalize() const {
  return UVec<S, N>::normalize(*this);
}

// Represents a ray
template <typename S, size_t N> class Ray {
public:
  Vec<S, N> orig; // origin of the ray
  UVec<S, N> dir; // direction of the ray

  Ray<S, N>() : dir(Vec<S, N>{1}.normalize()) {}

  Ray<S, N>(const Vec<S, N> &orig, const UVec<S, N> &dir)
      : orig(orig), dir(dir) {}

  // Printing ray
  inline friend std::ostream &operator<<(std::ostream &os, const Ray<S, N> &r) {
    return os << "<Ray " << r.orig << " " << r.dir << ">";
  }

  // Returns the length of the vector v projected onto the line associated
  // to the ray.
  inline S relativeLength(const Vec<S, N> &v) const {
    return this->dir.dot(v - this->orig);
  }

  // Returns whether the given vector is in view of the ray.
  inline bool inView(const Vec<S, N> &v) const {
    return this->relativeLength(v) >= 0;
  }

  // Project the vector v onto the line associated to the ray.
  inline Vec<S, N> project(const Vec<S, N> &v) const {
    return this->follow(this->relativeLength(v));
  }

  // Return the vector by following the ray for the given distance
  inline Vec<S, N> follow(S distance) const {
    return this->orig + (this->dir * distance);
  }
};

// Represents a 2-dimensional camera
template <typename S, size_t N> class Camera {
public:
  Camera(const Vec<S, N> &origin, const Vec<S, N> &centre,
         const Vec<S, N> &down, const Vec<S, N> &right, unsigned int hRes,
         unsigned int vRes)
      : origin(origin), centre(centre), down(down), right(right), hRes(hRes),
        vRes(vRes) {}

  Vec<S, N> origin;
  Vec<S, N> centre;
  Vec<S, N> down;
  Vec<S, N> right;
  unsigned int hRes;
  unsigned int vRes;
};

// The result of a ray hitting a body
template <typename S, size_t N> class Hit {
public:
  Hit(const Ray<S, N> &ray) : ray(ray) {}

  Ray<S, N> ray;
  S distance;
  const Body<S, N> *body;
  Vec<S, N> intercept;
};

// A body
template <typename S, size_t N> class Body {
public:
  // Returns whether the given ray hits the body.  If so, it fills out the
  // details in the hit structure.
  virtual bool intersect(const Ray<S, N> &ray, Hit<S, N> &hit) const = 0;

  virtual Interaction<S, N> next(const Hit<S, N> &hit) const = 0;
};

// Represents an interaction with a ray and an object: a convex combination
//
//     lambda |ray> +  (1 - lambda) |colour>,
//
// where ray represents the reflected/refracted/etc ray and colour the
// "absorbed" part.
template <typename S, size_t N> class Interaction {
public:
  Interaction(Colour<S> colour) : lambda(0), colour(colour), ray() {}
  Interaction(Ray<S, N> ray) : lambda(1), colour(), ray(ray) {}
  Interaction(S lambda, Ray<S, N> ray, Colour<S> colour)
      : lambda(lambda), colour(colour), ray(ray) {}

  S lambda;
  Colour<S> colour;
  Ray<S, N> ray;
};

// A scene
template <typename S, size_t N> class Scene final : public Body<S, N> {
public:
  std::vector<const Body<S, N> *> bodies;

  Scene<S, N>() : bodies{} {}

  Scene<S, N>(const std::initializer_list<const Body<S, N> *> &l) : bodies(l) {}

  bool intersect(const Ray<S, N> &ray, Hit<S, N> &minDistHit) const override {
    S minDist = std::numeric_limits<S>::infinity();
    Hit<S, N> hit(ray);
    bool ok = false;

    for (const Body<S, N> *body : bodies) {
      if (!body->intersect(ray, hit))
        continue;
      if (hit.distance < minDist && hit.distance > eps<S>) {
        minDist = hit.distance;
        minDistHit = hit;
        ok = true;
      }
    }

    return ok;
  }

  Interaction<S, N> next(const Hit<S, N> &hit) const override { assert(0); }
};

// A reflective sphere
template <typename S, size_t N>
class ReflectiveSphere final : public Body<S, N> {
public:
  Vec<S, N> centre;
  S radius;

  ReflectiveSphere(const Vec<S, N> &centre, S radius)
      : centre(centre), radius(radius) {}

  bool intersect(const Ray<S, N> &ray, Hit<S, N> &hit) const override {
    Vec<S, N> projCentre = ray.project(centre);

    if (!ray.inView(projCentre))
      return false;

    hit.body = this;

    Vec<S, N> aVec = projCentre - centre;
    S a = aVec.length();

    if (a >= radius)
      return false;

    S b = std::sqrt(radius * radius - a * a);
    hit.distance = (projCentre - ray.orig).length() - b;
    hit.intercept = hit.ray.follow(hit.distance);

    return true;
  }

  Interaction<S, N> next(const Hit<S, N> &hit) const override {
    UVec<S, N> normal = (centre - hit.intercept).normalize();
    UVec<S, N> dir = hit.ray.dir.reflect(normal);
    return Interaction<S, N>(Ray<S, N>(hit.intercept, dir));
  }
};

// Finds the least positive x such that both a1 <= x <= a2
// and b1 <= x <= b2.  Assumes a1 <= a2 and b1 <= b2.
template <typename S>
bool leastPositiveIntersection(S a1, S a2, S b1, S b2, S &x) {
  if (b2 < 0 || a2 < 0)
    return false;
  // case 1: [  a  ]
  //                  [   b   ]    or vice versa
  if (a2 < b1 || b2 < a1)
    return false;
  // case 2: [      a    ]
  //                [  b   ??
  if (a1 <= b1) {
    if (b1 >= 0)
      x = b1;
    else
      x = 0; // we know a2, b2 >= 0
    return true;
  }
  // case 2': [      b   ]
  //                 [  a  ??
  if (a1 >= 0)
    x = a1;
  else
    x = 0;
  return true;
}

// A reflective torus
template <typename S, size_t N>
class ReflectiveTorus final : public Body<S, N> {
  Vec<S, N> centre;
  S radius;

  Vec<S, N> centre1;
  Vec<S, N> centre2;
  Vec<S, N> P1;
  Vec<S, N> P2;

public:
  ReflectiveTorus(const Vec<S, N> &centre, S radius)
      : centre(centre), radius(radius), P1{0}, P2{0} {
    P2[0] = 1;
    P1[1] = 1;
    for (size_t i = 2; i < N; i++) {
      if ((i & 1) == 0)
        P1[i] = 1;
      else
        P2[i] = 1;
    }
    centre1 = pwProd(centre, P1);
    centre2 = pwProd(centre, P2);
  }

  bool intersect(const Ray<S, N> &ray, Hit<S, N> &hit) const override {
    Ray<S, N> ray1(pwProd(ray.orig, P1), pwProd(ray.dir, P1).normalize());
    Ray<S, N> ray2(pwProd(ray.orig, P2), pwProd(ray.dir, P2).normalize());
    S speed1 = pwProd(ray.dir, P1).length();
    S speed2 = pwProd(ray.dir, P2).length();
    Vec<S, N> projCentre1(ray1.project(centre1));
    Vec<S, N> projCentre2(ray2.project(centre2));

    S a1 = (projCentre1 - centre1).length();
    if (a1 >= radius)
      return false;

    S a2 = (projCentre2 - centre2).length();
    if (a2 >= radius)
      return false;

    S b1 = std::sqrt(radius * radius - a1 * a1);
    S b2 = std::sqrt(radius * radius - a2 * a2);

    S d1 = (projCentre1 - ray1.orig).length();
    S d2 = (projCentre2 - ray2.orig).length();

    if (!leastPositiveIntersection<S>((d1 - b1) / speed1, (d1 + b1) / speed1,
                                      (d2 - b2) / speed2, (d2 + b2) / speed2,
                                      hit.distance))
      return false;

    if (hit.distance < eps<S>)
      return false;

    hit.body = this;
    hit.intercept = ray.follow(hit.distance);
    return true;
  }

  Interaction<S, N> next(const Hit<S, N> &hit) const override {
    Vec<S, N> intercept1(pwProd(hit.intercept, P1));
    Vec<S, N> intercept2(pwProd(hit.intercept, P2));
    Vec<S, N> diff1 = intercept1 - centre1;
    Vec<S, N> diff2 = intercept2 - centre2;
    Vec<S, N> normal;
    if (diff1.length() > diff2.length())
      normal = diff1;
    else
      normal = diff2;
    UVec<S, N> dir = hit.ray.dir.reflect(normal.normalize());
    return Interaction<S, N>(Ray<S, N>(hit.intercept, dir));
  }
};

// Hyper-checkerboard
template <typename S, size_t N>
class HyperCheckerboard final : public Body<S, N> {
  Ray<S, N> normal;
  std::array<Vec<S, N>, N> axes;
  std::array<Ray<S, N>, N> axisRays;
  std::array<S, N> axisLengths;

public:
  HyperCheckerboard(const Ray<S, N> &normal,
                    const std::array<Vec<S, N>, N> &axes)
      : normal(normal), axes(axes) {
    for (size_t i = 0; i < N; i++) {
      axisRays[i] = Ray<S, N>(normal.orig, axes[i].normalize());
      axisLengths[i] = axes[i].length();
    }
  }

  bool intersect(const Ray<S, N> &ray, Hit<S, N> &hit) const override {
    hit.body = this;

    S offset = normal.relativeLength(ray.orig);
    S dir = normal.dir.dot(ray.dir);

    if (std::fabs(dir) <= eps<S>)
      return false;

    hit.distance = -offset / dir;

    if (hit.distance < 0)
      return false;

    hit.intercept = ray.follow(hit.distance);

    if (hit.intercept.length() > 15)
      return false;

    return true;
  }

  Interaction<S, N> next(const Hit<S, N> &hit) const override {
    int sum = 0;
    Colour<S> colour;

    for (size_t i = 0; i < N; i++) {
      auto t = axisRays[i].relativeLength(hit.intercept) / axisLengths[i];
      sum += (int)std::fabs(std::floor(t));
    }

    return Interaction<S, N>(
        0.2, Ray<S, N>(hit.intercept, hit.ray.dir.reflect(normal.dir)),
        sum % 2 ? black<S> : white<S>);
  }
};

template <typename S> class PNGScreen {
public:
  png::image<png::rgb_pixel> png;
  PNGScreen(size_t hRes, size_t vRes) : png(hRes, vRes) {}

  inline void put(size_t x, size_t y, const Colour<S> &colour) {
    png[x][y] = colour;
  }
};

template <typename S, size_t N, typename SCREEN, typename RND = std::mt19937>
class Sampler {
  struct Job {
    const size_t start;
    const size_t end;
    std::vector<Colour<S>> result;

    Job(size_t start, size_t end) : start(start), end(end) {
      result.reserve(end - start);
    }
  };

  const Body<S, N> &root;
  const Camera<S, N> &camera;
  SCREEN &screen;

  int maxBounces;
  int firstBatch;
  S target;
  size_t pixelsPerJob;

  std::mutex lock;
  std::condition_variable done;

  std::vector<std::unique_ptr<std::thread>> workers;
  std::vector<Job> jobs;
  size_t nextJob;

public:
  size_t nWorkers = 0;

  Sampler(const Body<S, N> &root, const Camera<S, N> &camera, SCREEN &screen)
      : root(root), camera(camera), screen(screen), maxBounces(20),
        firstBatch(10), target(.05), pixelsPerJob(500) {}

  // Shoots the scene with the camera provided and writes out to the screen.
  // Can be called only once.
  void shoot() {
    {
      std::unique_lock<std::mutex> g(lock);

      // Start threads
      if (nWorkers == 0)
        nWorkers = std::thread::hardware_concurrency();
      workers.reserve(nWorkers);
      for (size_t i = 0; i < nWorkers; i++)
        workers.push_back(
            std::make_unique<std::thread>(&Sampler::workerEntry, this));

      // Create jobs.  Thread will be waiting on lock until we release it.
      size_t nPixels = camera.vRes * camera.hRes;
      for (size_t job = 0; job < nPixels; job += pixelsPerJob)
        jobs.emplace_back(job, std::min(job + pixelsPerJob, nPixels));
      nextJob = 0;
    }

    {
      std::unique_lock<std::mutex> g(lock);
      while (true) {
        if (done.wait_for(g, std::chrono::milliseconds(100))
              == std::cv_status::no_timeout)
          break;
        std::cerr << nextJob << "/" << jobs.size() << "\n";
      }
    }

    for (auto &thread : workers)
      thread->join();

    std::cerr << "\nWriting to PNG ...\n";
    for (auto &job : jobs) {
      size_t j = 0;
      for (size_t i = job.start; i < job.end; i++, j++) {
        size_t x = i % camera.vRes; // TODO optimize?
        size_t y = i / camera.vRes;
        screen.put(x, y, job.result[j]);
      }
    }
  }

private:
  void workerEntry() {
    RND rnd;

    while (true) {
      size_t ourJob;

      { // get next job
        std::unique_lock<std::mutex> g(lock);
        if (nextJob == jobs.size())
          break;
        ourJob = nextJob++;
        if (nextJob == jobs.size())
          done.notify_all();
      }

      Job &job = jobs[ourJob];

      for (size_t i = job.start; i < job.end; i++) {
        size_t x = i % camera.vRes; // TODO optimize?
        size_t y = i / camera.vRes;
        Vec<S, N> down(camera.down * (2 * static_cast<S>(x) - camera.vRes) / 2);
        Vec<S, N> right(camera.right * (2 * static_cast<S>(y) - camera.hRes) /
                        2);
        Ray<S, N> ray(camera.origin,
                      (down + right + camera.centre).normalize());
        Colour<S> c(sample(ray, camera.right, camera.down, rnd));
        job.result.push_back(c);
      }
    }
  }

  inline Colour<S> sampleOne(const Ray<S, N> &r, const Vec<S, N> &dx,
                             const Vec<S, N> &dy, RND &rnd) const {
    std::uniform_real_distribution<S> rnd01(0, 1);
    Vec<S, N> jitter(dx * rnd01(rnd) + dy * rnd01(rnd));
    Ray<S, N> cRay(r.orig, (r.dir + jitter).normalize());

    S factor = 1;
    Colour<S> ret;

    for (int i = 0; i < maxBounces; i++) {
      Hit<S, N> hit(cRay);
      if (!root.intersect(cRay, hit))
        return ret;
      Interaction<S, N> intr = hit.body->next(hit);
      if (intr.lambda == 0)
        return ret + intr.colour * factor;
      ret += intr.colour * factor * (1 - intr.lambda);
      factor *= intr.lambda;
      if (factor < this->target)
        return ret;
      cRay = intr.ray;
    }

    // max bounces hit.
    std::cerr << "m";
    std::cerr.flush();
    return ret;
  }

  inline Colour<S> sampleBatch(const Ray<S, N> &ray, unsigned int size,
                               const Vec<S, N> &dx, const Vec<S, N> &dy,
                               RND &rnd) const {
    Colour<S> ret = black<S>;
    for (unsigned int i = 0; i < size; i++)
      ret += sampleOne(ray, dx, dy, rnd);
    return ret / static_cast<S>(size);
  }

  inline Colour<S> sample(const Ray<S, N> &ray, const Vec<S, N> &dx,
                          const Vec<S, N> &dy, RND &rnd) const {
    Colour<S> oldCol;
    Colour<S> newCol;

    unsigned int size = this->firstBatch;

    oldCol = sampleBatch(ray, size, dx, dy, rnd);
    newCol = sampleBatch(ray, size, dx, dy, rnd);

    for (;;) {
      bool done = (oldCol - newCol).supNorm() < this->target;
      oldCol = (oldCol + newCol) / 2;
      size = 2 * size;

      if (done)
        return oldCol;

      newCol = sampleBatch(ray, size, dx, dy, rnd);
    }
  }
};

template <typename S, size_t N> void render(size_t nWorkers) {
  constexpr int dpi = 150;
  constexpr int hRes = static_cast<int>(6.81102 * static_cast<double>(dpi));

  // Ratio of front cover is 173 : 246
  constexpr int vRes = static_cast<int>(
      (static_cast<double>(hRes) / 173.) * 246.);

  std::cerr << "Rendering @" << dpi << " DPI (" << hRes << "x" << vRes << ")\n";

  Vec<S, N> origin{0};
  Vec<S, N> e0{1};

  // ReflectiveSphere<S,N> sphere(origin, .9);
  ReflectiveTorus<S, N> torus(origin, .636);
  std::array<Vec<S, N>, N> floorAxes;
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      floorAxes[i][j] = (j == i + 1) ? 1 : 0;
  HyperCheckerboard<S, N> floor(Ray<S, N>(e0, (-e0).normalize()), floorAxes);
  HyperCheckerboard<S, N> ceiling(Ray<S, N>((-e0), e0.normalize()), floorAxes);
  // Scene<S,N> scene{&sphere, &floor, &ceiling};
  Scene<S, N> scene{&torus, &floor, &ceiling};

  Camera<S, N> camera(Vec<S, N>{0, -2, -2}, // origin
                      Vec<S, N>{},          // centre
                      Vec<S, N>{-1},        // down
                      Vec<S, N>{0, 1, -1},  // right
                      hRes, vRes);
  camera.centre = camera.origin * -.4;

  for (size_t i = 0; i < (N - 3) / 2; i++) {
    camera.right[3 + 2 * i] = std::pow(2, -static_cast<S>(i) - 1);
    camera.down[4 + 2 * i] = std::pow(2, -static_cast<S>(i) - 1);
  }

  constexpr auto minRes = std::min(hRes, vRes);

  camera.right = camera.right.normalize() / minRes;
  camera.down = camera.down.normalize() / minRes;

  PNGScreen<S> screen(camera.hRes, camera.vRes);
  Sampler<S, N, PNGScreen<S>> sampler(scene, camera, screen);
  sampler.nWorkers = nWorkers;
  sampler.shoot();
  screen.png.write("out.png");
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
  render<double, 5>(nWorkers);
  return 0;
}

// vim: sw=2 ts=2 et
