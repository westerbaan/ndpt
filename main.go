package ndpt

import (
	"log"
	"math"
)

type Vector []float64
type UnitVector Vector

const eps float64 = 1e-9

type Color struct {
	R, G, B float64
}

var Black Color = Color{0, 0, 0}
var White Color = Color{1, 1, 1}

type Ray struct {
	Origin    Vector
	Direction UnitVector
}

func (v Vector) Add(w Vector) Vector {
	ret := make([]float64, len(v), len(v))
	for i := 0; i < len(v); i++ {
		ret[i] = v[i] + w[i]
	}
	return ret
}

func (v Vector) Sub(w Vector) Vector {
	ret := make([]float64, len(v), len(v))
	for i := 0; i < len(v); i++ {
		ret[i] = v[i] - w[i]
	}
	return ret
}

func (v Vector) Dot(w Vector) (res float64) {
	for i := 0; i < len(v); i++ {
		res += v[i] * w[i]
	}
	return
}

func (v Vector) Length() float64 {
	return math.Sqrt(v.Dot(v))
}

func (v Vector) Scale(scalar float64) Vector {
	ret := make([]float64, len(v), len(v))
	for i := 0; i < len(v); i++ {
		ret[i] = v[i] * scalar
	}
	return ret
}

func (v Vector) Normalize() UnitVector {
	return UnitVector(v.Scale(1.0 / v.Length()))
}

func (v UnitVector) Dot(w Vector) float64 {
	return Vector(v).Dot(w)
}

func (v UnitVector) Scale(scalar float64) {
	return Vector(v).Scale(scalar)
}

// Returns the length of the vector projected onto the line associated with the
// ray relative to the origin of the ray.
func (r Ray) RelativeLength(v Vector) float64 {
	return v.Sub(r.Origin).Dot(Vector(r.Direction))
}

// Returns the vector v projected onto the line associated with the ray
func (r Ray) Project(v Vector) Vector {
	return Vector(r.Direction).Scale(r.RelativeLength(v)).Add(r.Origin)
}

func (r Ray) Follow(distance float64) Vector {
	return r.Origin.Add(r.Direction.Scale(distance))
}

type Hit interface {
	Distance() float64 // distance at which it hit; infty if it didn't

	// If hit, computes what happens next.  Returns a Color if the ray is
	// emitted/absorped or a Ray if it bounced/reflected/refracted/....
	Next() (*Ray, *Color)
}

type Body interface {
	Intersect(Ray) Hit
}

type Scene struct {
	Bodies []Body
}

// Scene currently simply checks all bodies for a hit.  For a bigger scenes,
// we might want to add bounding boxes to Body's and add a tree.
func (s *Scene) Intersect(ray Ray) Hit {
	var minDistHit Hit
	minDist := math.Inf(1)
	for _, body := range s.Bodies {
		hit := body.Intersect(ray)
		dist := hit.Distance()
		if dist < minDist {
			minDist = dist
			minDistHit = hit
		}
	}
	return minDistHit
}

type Sampler struct {
	Root       Body
	MaxBounces int
}

func (s *Sampler) Sample(ray Ray) Color {
	body := s.Root
	for i := 0; i < s.MaxBounces; i++ {
		hit := body.Intersect(ray)
		dist := hit.Distance()
		if math.IsInf(dist, 1) {
			return Black
		}
		rayPtr, color := hit.Next()
		if color != nil {
			return *color
		}
		ray = *rayPtr
	}
	log.Print("MaxBounces hit :(")
	return Black
}

type ReflectiveSphere struct {
	Origin Vector
	Radius float64
}

type ReflectiveSphereHit struct {
	Sphere *ReflectiveSphere
}

func (b *ReflectiveSphere) Intersect(ray Ray) Hit {

}

type HyperCheckerboard struct {
	Normal Ray
	Axes   []Vector
}

func (b *HyperCheckerboard) Intersect(ray Ray) Hit {
	offset := b.Normal.RelativeLength(ray.Origin)
	direction := b.Normal.Direction.Dot(Vector(ray.Direction))

	if math.Abs(direction) <= eps {
		// ray is parallel to hyperplane - not hit
		return nil
	}

	distance = -offset / direction

	if distance < 0 {
		// ray moved away from the hyperplane - not hit
		return nil
	}

	return &hcbHit{ray, b, distance}
}

type hcbHit struct {
	ray      Ray
	board    *HyperCheckerboard
	distance float64
}

func (h *hcbHit) Distance() float64 {
	return h.distance
}

func (h *hcbHit) Next() (ray *Ray, color *Color) {
	intercept := h.ray.Follow(h.distance)
	t := make([]float64, len(h.board.Axes))

	for i := 0; i < len(t); i++ {
		axisray := Ray{board.Origin, board.Axes[i].Normalize()}
		t[i] = axisray.RelativeLength(intercept)
		t[i] = t[i] / board.Axes[i].Length()
	}

	sum := 0
	for i := 0; i < len(t); i++ {
		sum += int(t[i])
	}
	sign := sum % 2

	if sign == 1 {
		colour = &Black
	} else {
		colour = &White
	}
}
