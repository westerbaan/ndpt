#include <cmath>
#include <cassert>

#include <array>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#include <png++/png.hpp>

template <class S>
constexpr S eps = std::numeric_limits<S>::epsilon();

template <class S, size_t N>
class uvec; // unit vector, see below

template <class S, size_t N>
class body;

template <class S, size_t N>
class interaction;

template <class S>
class colour {
public:
    S R, G, B;

    constexpr colour(const S R, const S G, const S B) : R(R), G(G), B(B) { }
    colour() : R(0), G(0), B(0) { }

    // Adds two colours
    inline colour<S>&
    operator+=(const colour<S>& rhs) {
        this->R += rhs.R; this->G += rhs.G; this->B += rhs.B;
        return *this;
    }
    inline const colour<S>
    operator+(const colour<S>& rhs) const {
        return colour<S>(*this) += rhs;
    }

    // Substracts colours
    inline colour<S>&
    operator-=(const colour<S>& rhs) {
        this->R -= rhs.R; this->G -= rhs.G; this->B -= rhs.B;
        return *this;
    }
    inline const colour<S>
    operator-(const colour<S>& rhs) const {
        return colour<S>(*this) -= rhs;
    }

    // Scalar multiplication
    inline const colour<S>
    operator*=(const S& s) {
        this->R *= s; this->G *= s; this->B *= s;
        return *this;
    }
    inline const colour<S>
    operator*(const S& rhs) const {
        return colour<S>(*this) *= rhs;
    }
    inline friend const colour<S>
    operator*(const S& lhs, const colour<S>& rhs) {
        return colour<S>(rhs) *= lhs;
    }
    inline const colour<S> operator/(const S& rhs) const {
        return *this * (1/rhs);
    }

    // Supremum norm
    inline S supNorm() const {
        return std::max(this->R, std::max(this->G, this->B));
    }

    // Implicit casts
    inline operator png::rgb_pixel() const {
        return png::rgb_pixel(R * 255, G * 255, B * 255);
    }

    // Printing colour
    inline friend std::ostream&
    operator<<(std::ostream& os, const colour<S>& col) {
        return os << '(' << col.R << ", " << col.G << ", " << col.B << ')';
    }
};

template <class S>
constexpr colour<S> black = colour<S>(0,0,0);

template <class S>
constexpr colour<S> white = colour<S>(1,1,1);

template <class S>
constexpr colour<S> red = colour<S>(1,0,0);


// Represents a vector of N-elements with scalar of type S
template <class S, size_t N>
class vec {
protected:
    // Components of the vector
    std::array<S, N> vals;

public:
    // Uninitialized vector
    vec<S,N>() { }

    // Copy constructor
    vec<S,N>(const vec<S,N> &other) = default;

    // Initializer list constructor
    template<class T>
    vec<S,N>(const std::initializer_list<T> &l) {
        // For some reason clang doesn't think l.size() is constexpr
        // static_assert(l.size() < N, "initializer_list too long");
        assert(l.size() < N);
        std::copy(l.begin(), l.end(), this->vals.begin());
        std::fill(&this->vals[l.size()], this->vals.end(), 0);
    }

    // Adds the given vector to this one.
    inline vec<S,N>&
    operator+=(const vec<S,N>& rhs) {
        for (size_t i = 0; i < N; i++) this->vals[i] += rhs.vals[i];
        return *this;
    }

    // Adds two vectors
    inline const vec<S,N>
    operator+(const vec<S,N>& rhs) const {
        return vec<S,N>(*this) += rhs;
    }

    // Substracts the given vector from this one.
    inline vec<S,N>&
    operator-=(const vec<S,N>& rhs) {
        for (size_t i = 0; i < N; i++) this->vals[i] -= rhs.vals[i];
        return *this;
    }

    // Substracts two vectors
    inline const vec<S,N>
    operator-(const vec<S,N>& rhs) const {
        return vec<S,N>(*this) -= rhs;
    }

    // Inverts vector
    inline const vec<S,N>
    operator-() const {
        vec<S,N> ret;
        for (size_t i = 0; i < N; i++) ret.vals[i] = -this->vals[i];
        return ret;
    }

    // Scalar multiplication in-place
    inline const vec<S,N>
    operator*=(const S& s) {
        for (size_t i = 0; i < N; i++) this->vals[i] *= s;
        return *this;
    }

    // Scalar multiplication on the right
    inline const vec<S,N>
    operator*(const S& rhs) const {
        return vec<S,N>(*this) *= rhs;
    }

    // Scalar multiplication on the left
    inline friend const vec<S,N>
    operator*(const S& lhs, const vec<S,N>& rhs) {
        return vec<S,N>(rhs) *= lhs;
    }

    // Division scalar multiplication
    inline const vec<S,N> operator/(const S& rhs) const {
        return *this * (1/rhs);
    }

    // Printing vectors
    inline friend std::ostream&
    operator<<(std::ostream& os, const vec<S,N>& vec) {
        os << '[';
        if (N != 0) os << vec.vals[0];
        for (size_t i = 1; i < N; i++) os << ' ' <<  vec.vals[i];
        os << ']'; return os;
    }

    // Compute inner product
    inline S
    dot (const vec<S,N>& other) const {
        S ret = 0;
        for (int i = 0; i < N; i++) ret += this->vals[i] * other.vals[i];
        return ret;
    }

    // Length of vector
    inline S length() const { return std::sqrt(this->dot(*this)); }

    // Subscript operator
    inline S& operator[] (const size_t i) {
        return vals[i];
    }

    const uvec<S,N> normalize() const;
};

// Represents a unit-vector vector of N-elements with scalar of type S
template <class S, size_t N>
class uvec {
    vec<S,N> v;
    uvec<S,N>() { }

public:
    // Create a unit vector by normalizing a regular vector
    inline static const uvec<S,N>
    normalize(const vec<S,N>& v) {
        uvec<S,N> ret;
        ret.v = v / v.length();
        return ret;
    }

    // Reflect this vector over the given normal
    inline uvec<S,N>
    reflect(const uvec<S,N> &normal) const {
        return (*this - normal * (2 * this->dot(normal))).normalize();
    }

    // Cast to vector
    inline operator vec<S,N>() const {
        return this->v;
    }

    // Implicit casts and templates don't combine well, so we have to add
    // quite some boilerplate to make the compiler understand uv1 + uv2
    // actually means (vec)uv1 + (vec)uv2.
    inline S dot (const vec<S,N>& other) const {
        return static_cast<vec<S,N>>(*this).dot(other);
    }
    inline const vec<S,N> operator*(const S& rhs) const {
        return vec<S,N>(*this) *= rhs;
    }
    inline const vec<S,N> operator/(const S& rhs) const {
        return static_cast<vec<S,N>>(*this) / rhs;
    }
    inline const vec<S,N> operator-(const vec<S,N>& rhs) const {
        return static_cast<vec<S,N>>(*this) - rhs;
    }
    inline const vec<S,N> operator-(const uvec<S,N>& rhs) const {
        return *this - static_cast<vec<S,N>>(rhs);
    }
    inline const vec<S,N> operator-() const {
        return -static_cast<vec<S,N>>(*this);
    }
    inline const vec<S,N> operator+(const vec<S,N>& rhs) const {
        return static_cast<vec<S,N>>(*this) + rhs;
    }
    inline const vec<S,N> operator+(const uvec<S,N>& rhs) const {
        return *this + static_cast<vec<S,N>>(rhs);
    }
    inline friend std::ostream&
    operator<<(std::ostream& os, const uvec<S,N>& v) {
        return os << static_cast<vec<S,N>>(v);
    }
};

// Normalize vector
template <class S, size_t N>
inline const uvec<S,N>
vec<S,N>::normalize() const { return uvec<S,N>::normalize(*this); }

// Represents a ray
template <class S, size_t N>
class ray {
public:
    vec<S,N> orig; // origin of the ray
    uvec<S,N> dir; // direction of the ray

    ray<S,N>() : dir(vec<S,N>{1}.normalize()) { }

    ray<S,N>(const vec<S,N> &orig, const uvec<S,N> &dir)
                    : orig(orig), dir(dir) { }

    // Printing ray
    inline friend std::ostream&
    operator<<(std::ostream& os, const ray<S,N>& r) {
        return os << "<Ray " << r.orig << " " << r.dir << ">";
    }

    // Returns the length of the vector v projected onto the line associated
    // to the ray.
    inline S
    relativeLength(const vec<S,N> &v) const {
        return this->dir.dot(v - this->orig);
    }

    // Returns whether the given vector is in view of the ray.
    inline bool
    inView(const vec<S,N> &v) const {
        return this->relativeLength(v) >= 0;
    }

    // Project the vector v onto the line associated to the ray.
    inline vec<S,N>
    project(const vec<S,N> &v) const {
        return this->follow(this->relativeLength(v));
    }

    // Return the vector by following the ray for the given distance
    inline vec<S,N>
    follow(S distance) const {
        return this->orig + (this->dir * distance);
    }
};

// Represents a 2-dimensional camera
template <class S, size_t N>
class camera {
public:
    camera(const vec<S,N> &origin, const vec<S,N> &centre, 
            const vec<S,N> &down, const vec<S,N> &right,
            unsigned int hRes, unsigned int vRes)
        : origin(origin), centre(centre), down(down), right(right),
            hRes(hRes), vRes(vRes) { }

    vec<S,N> origin;
    vec<S,N> centre;
    vec<S,N> down;
    vec<S,N> right;
    unsigned int hRes;
    unsigned int vRes;
};

// The result of a ray hitting a body
template <class S, size_t N>
class hit {
public:
    hit(const ray<S,N> &ray) : ray(ray) { }

    ray<S,N> ray;
    S distance;
    const body<S,N>* body;
    vec<S,N> intercept;
};

// A body
template <class S, size_t N>
class body {
public:
    // Returns whether the given ray hits the body.  If so, it fills out the
    // details in the hit structure.
    virtual bool
    intersect(const ray<S,N> &ray, hit<S,N>& hit) const = 0;

    virtual interaction<S,N>
    next(const hit<S,N>& hit) const = 0;
};

// Represents an interaction with a ray and an object: a convex combination
// 
//     lambda |ray> +  (1 - lambda) |colour>,
//
// where ray represents the reflected/refracted/etc ray and colour the
// "absorbed" part.
template <class S, size_t N>
class interaction {
public:
    interaction(colour<S> colour) : lambda(0), colour(colour), ray() { }
    interaction(ray<S,N> ray) : lambda(1), colour(), ray(ray) { }
    interaction(S lambda, ray<S,N> ray, colour<S> colour)
            : lambda(lambda), colour(colour), ray(ray) { }

    S lambda;
    colour<S> colour;
    ray<S,N> ray;
};

// A scene
template <class S, size_t N>
class scene final : public body<S,N> {
public:
    std::vector<const body<S,N>*> bodies;

    scene<S,N>() : bodies{} { }

    scene<S,N>(const std::initializer_list<const body<S,N>*> &l) : bodies(l) { }

    bool
    intersect(const ray<S,N> &ray, hit<S,N>& minDistHit) const override {
        S minDist = std::numeric_limits<S>::infinity();
        hit<S,N> hit(ray);
        bool ok = false;

        for (const body<S,N>* body : bodies) {
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

    interaction<S,N>
    next(const hit<S,N>& hit) const override {
        assert(0);
    }
};

// A reflective sphere
template <class S, size_t N>
class reflectiveSphere final : public body<S,N> {
public:
    vec<S,N> centre;
    S radius;

    reflectiveSphere(vec<S,N> centre, S radius)
        : centre(centre), radius(radius) { }

    bool
    intersect(const ray<S,N> &ray, hit<S,N>& hit) const override {
        vec<S,N> projCentre = ray.project(centre);

        if (!ray.inView(projCentre)) return false;

        hit.body = this;

        vec<S,N> aVec = projCentre - centre;
        S a = aVec.length();

        if (a >= radius) return false;

        S b = std::sqrt(radius * radius - a*a);
        hit.distance = (projCentre - ray.orig).length() - b;
        hit.intercept = hit.ray.follow(hit.distance);

        return true;
    }

    interaction<S,N>
    next(const hit<S,N>& hit) const override {
        uvec<S,N> normal = (centre - hit.intercept).normalize();
        uvec<S,N> dir = hit.ray.dir.reflect(normal);
        return interaction<S,N>(ray<S,N>(hit.intercept, dir));
    }
};


// Hyper-checkerboard
template <class S, size_t N>
class hyperCheckerboard final : public body<S,N> {
    ray<S,N> normal;
    std::array<vec<S,N>,N> axes;
    std::array<ray<S,N>,N> axisRays;
    std::array<S,N> axisLengths;

public:
    hyperCheckerboard(const ray<S,N> &normal, const std::array<vec<S,N>,N> &axes)
            : axes(axes), normal(normal) {
        for (size_t i = 0; i < N; i++) {
            axisRays[i] = ray<S,N>(normal.orig, axes[i].normalize());
            axisLengths[i] = axes[i].length();
        }
    }

    bool
    intersect(const ray<S,N> &ray, hit<S,N>& hit) const override {
        hit.body = this;

        S offset = normal.relativeLength(ray.orig);
        S dir = normal.dir.dot(ray.dir);

        if (std::fabs(dir) <= eps<S>) return false;

        hit.distance = -offset / dir;

        if (hit.distance < 0) return false;

        hit.intercept = ray.follow(hit.distance);

        if (hit.intercept.length() > 15) return false;

        return true;
    }

    interaction<S,N>
    next(const hit<S,N>& hit) const override {
        int sum = 0;
        colour<S> colour;

        for (size_t i = 0; i < N; i++ ) {
            auto t = axisRays[i].relativeLength(hit.intercept) / axisLengths[i];
            sum += (int)std::fabs(std::floor(t));
        }

        return interaction<S,N>(
                0.2,
                ray<S,N>(hit.intercept, hit.ray.dir.reflect(normal.dir)),
                sum % 2 ? black<S> : white<S>);
    }
};

template <class S, size_t N, class RND=std::mt19937>
class sampler {
public:
    const body<S,N>& root;
    int maxBounces;
    int firstBatch;
    S target;

    sampler(const body<S,N>& root)
        : root(root), maxBounces(20), firstBatch(10), target(.05) { }

    inline colour<S> sampleOne(const ray<S,N> &r, const vec<S,N> &dx,
                               const vec<S,N> &dy, RND &rnd) const {
        std::uniform_real_distribution<S> rnd01(0,1);
        vec<S,N> jitter(dx * rnd01(rnd) + dy * rnd01(rnd));
        ray<S,N> cRay(r.orig, (r.dir + jitter).normalize());

        S factor = 1;
        colour<S> ret;

        for (int i = 0; i < maxBounces; i++) {
            hit<S,N> hit(cRay);
            if (!root.intersect(cRay, hit))
                return ret;
            interaction<S,N> intr = hit.body->next(hit);
            if (intr.lambda == 0)
                return ret + intr.colour * factor;
            ret += intr.colour * factor * (1 - intr.lambda);
            factor *= intr.lambda;
            if (factor < this->target)
                return ret;
            cRay = intr.ray;
        }

        // max bounces hit.
        std::cerr << "max bounces hit" << std::endl;
        return ret;
    }

    inline colour<S> sampleBatch(const ray<S,N> &ray,
                                 unsigned int size,
                                 const vec<S,N> &dx,
                                 const vec<S,N> &dy,
                                 RND &rnd) const {
        colour<S> ret = black<S>;
        for (unsigned int i = 0; i < size; i++)
            ret += sampleOne(ray, dx, dy, rnd);
        return ret / static_cast<S>(size);
    }

    inline colour<S> sample(const ray<S,N> &ray, const vec<S,N> &dx,
                            const vec<S,N> &dy, RND &rnd) const {
        colour<S> oldCol;
        colour<S> newCol;

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

    void shoot (const camera<S,N>& camera) {
        RND rnd;
        png::image<png::rgb_pixel> image(camera.hRes, camera.vRes);

        for (size_t x = 0; x < camera.vRes; x++) {
            if (x && x % 100 == 0) std::cout << " " << x << std::endl;
            for (size_t y = 0; y < camera.hRes; y++) {
                vec<S,N> down(camera.down * (2*static_cast<S>(x) - camera.vRes) / 2);
                vec<S,N> right(camera.right * (2*static_cast<S>(y) - camera.hRes) / 2);
                ray<S,N> ray(camera.origin, (down + right + camera.centre).normalize());
                colour<S> c(sample(ray, camera.right, camera.down, rnd));
                image[x][y] = c;
            }
        }

        image.write("out.png");
    }

};

template <class S, size_t N>
void render() {
    int hRes = 750;
    int vRes = 750;

    vec<S,N> origin;
    vec<S,N> e0{1};

    reflectiveSphere<S,N> sphere(origin, .9);
    std::array<vec<S,N>,N> floorAxes;
    for (size_t i = 0; i < N-1; i++) floorAxes[i][i+1] = 1;
    hyperCheckerboard<S,N> floor(ray<S,N>(e0, (-e0).normalize()), floorAxes);
    hyperCheckerboard<S,N> ceiling(ray<S,N>((-e0), e0.normalize()), floorAxes);
    // scene<S,N> scene{&sphere};
    scene<S,N> scene{&sphere, &floor, &ceiling};

    camera<S,N> camera(
            vec<S,N>{0, -2, -2}, // origin
            vec<S,N>{},          // centre
            vec<S,N>{-1},        // down
            vec<S,N>{0, 1, -1},  // right
            hRes,
            vRes);
    camera.centre = camera.origin * -.4;

    for (size_t i = 0; i < (N-3)/2; i++) {
        camera.right[3+2*i] = std::pow(2, -static_cast<S>(i)-1);
        camera.down[4+2*i] = std::pow(2, -static_cast<S>(i)-1);
    }

    camera.right = camera.right.normalize() / hRes;
    camera.down = camera.down.normalize() / vRes;

    sampler<S,N> sampler(scene);
    sampler.shoot(camera);
}

int main() {
    render<double,5>();
    return 0;
}
