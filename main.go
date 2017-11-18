package ndpt

import (
	"log"
	"math"
)

type Vector []float64

type Color struct {
	R, G, B float64
}

type Ray struct {
	Origin    Vector
	Direction Vector
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
	body := s.Body
	for i := 0; i < s.MaxBounces; i++ {
		hit := body.Intersect(ray)
		dist := hit.Distance()
		if math.IsInf(dist) {
			return // nothing hit: black
		}
		rayPtr, color := hit.Next()
		if color != nil {
			return *color
		}
		ray = *rayPtr
	}
	log.Print("MaxBounces hit :(")
	return
}

type Hypercheckerboard struct {
	Axes []Vector
}

func (b *Hypercheckerboard) Intersect(ray Ray) Hit {

}

type hcbHit struct {
}

func (hcbHit h) Distance() float64 {
}

func (hcbHit h) Next() (*Ray, *Color) {

}
