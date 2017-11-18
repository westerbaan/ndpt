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

// Returns whether the vector is almost zero, see eps.
func (v Vector) IsZero() bool {
	return v.Length() < eps
}

// Returns the length of the vector projected onto the line associated with the
// ray relative to the origin of the ray.
func (r Ray) RelativeLength(v Vector) float64 {
	return v.Sub(r.Origin).Dot(Vector(r.Direction))
}

func (r Ray) InView(v Vector) bool {
	return r.RelativeLength(v) >= 0
}

// Returns the vector v projected onto the line associated with the ray
func (r Ray) Project(v Vector) Vector {
	return Vector(r.Direction).Scale(r.RelativeLength(v)).Add(r.Origin)
}

type Hit interface {
	Distance() float64

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
		if hit == nil {
			continue
		}
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
		if hit == nil {
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
	Centre Vector
	Radius float64
}

type reflectiveSphereHit struct {
	Sphere   *ReflectiveSphere
	distance float64
}

func (h reflectiveSphereHit) Distance() float64 {
	return h.distance
}

func (h reflectiveSphereHit) Next() (*Ray, *Color) {
	return nil, nil
}

func (sphere *ReflectiveSphere) Intersect(ray Ray) Hit {
	var ret reflectiveSphereHit
	projCentre := ray.Project(sphere.Centre)

	if !ray.InView(projCentre) {
		return nil
	}

	a := projCentre.Sub(sphere.Centre).Length()
	b := math.Sqrt(sphere.Radius*sphere.Radius - a*a)
	ret.distance = projCentre.Sub(ray.Origin).Length()

	if a >= sphere.Radius {
		return nil
	}

	return &ret
}

type Hypercheckerboard struct {
	Axes []Vector
}

func (b *Hypercheckerboard) Intersect(ray Ray) Hit {

}

type hcbHit struct {
}

func (h hcbHit) Distance() float64 {
}

func (h hcbHit) Next() (*Ray, *Color) {

}
