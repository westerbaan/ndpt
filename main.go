package main

import (
	"image"
	"image/png"
	"log"
	"math"
	"os"
	"runtime"
)

type Vector []float64
type UnitVector Vector

const eps float64 = 1e-9

type Colour struct {
	R, G, B float64
}

// to implement the image.Color interface:
func (c Colour) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R * 0xffff)
	b = uint32(c.B * 0xffff)
	g = uint32(c.G * 0xffff)
	a = 0xffff
	return
}

var Black Colour = Colour{0, 0, 0}
var White Colour = Colour{1, 1, 1}

func (c Colour) Add(d Colour) Colour {
	return Colour{c.R + d.R, c.G + d.G, c.B + d.B}
}

func (c Colour) Sub(d Colour) Colour {
	return Colour{c.R - d.R, c.G - d.G, c.B - d.B}
}

func (c Colour) Scale(scalar float64) Colour {
	return Colour{scalar * c.R, scalar * c.G, scalar * c.B}
}

func (c Colour) SupNorm() float64 {
	return math.Max(math.Abs(c.R), math.Max(math.Abs(c.G), math.Abs(c.B)))
}

type Image struct {
	Height, Width int
	Pixels        [][]Colour
}

func (imag *Image) ToNRGBA() (nrgba *image.NRGBA) {
	nrgba = image.NewNRGBA(image.Rect(0, 0, imag.Width, imag.Height))
	for i := 0; i < imag.Height; i++ {
		for j := 0; j < imag.Width; j++ {
			nrgba.Set(j, i, imag.Pixels[i][j])
		}
	}
	return
}

type Ray struct {
	Origin    Vector
	Direction UnitVector
}

func V(vs ...float64) Vector {
	return Vector(vs)
}

// m-th standard basis vector in N dimensions
func E(N, m int) Vector {
	res := make([]float64, N)
	res[m] = 1
	return Vector(res)
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

func (v UnitVector) Dot(w Vector) float64 {
	return Vector(v).Dot(w)
}

func (v UnitVector) Scale(scalar float64) Vector {
	return Vector(v).Scale(scalar)
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

func (r Ray) Follow(distance float64) Vector {
	return r.Origin.Add(r.Direction.Scale(distance))
}

type Hit interface {
	Distance() float64

	// If hit, computes what happens next.  Returns a Colour if the ray is
	// emitted/absorped or a Ray if it bounced/reflected/refracted/....
	Next() (*Ray, *Colour)
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
	FirstBatch int
	Target     float64
	// if the difference between the avarage colours of two consecutive
	// batches is below Target, their avarage is accepted as
	// the result of the
}

func (s *Sampler) SampleOne(ray Ray) Colour {
	body := s.Root
	for i := 0; i < s.MaxBounces; i++ {
		hit := body.Intersect(ray)
		if hit == nil {
			return Black
		}
		rayPtr, colour := hit.Next()
		if colour != nil {
			return *colour
		}
		ray = *rayPtr
	}
	log.Print("MaxBounces hit :(")
	return Black
}

func (s *Sampler) SampleBatch(ray Ray, size int) (c Colour) {
	for i := 0; i < size; i++ {
		c = c.Add(s.SampleOne(ray))
	}
	c = c.Scale(1 / float64(size))
	return
}

func (s *Sampler) Sample(ray Ray) Colour {
	var old, new Colour

	size := s.FirstBatch

	old = s.SampleBatch(ray, size)
	new = s.SampleBatch(ray, size)

	for {
		if old.Sub(new).SupNorm() < s.Target {
			return old.Add(new).Scale(.5)
		}

		old = old.Add(new).Scale(.5)
		size = 2 * size
		new = s.SampleBatch(ray, size)
	}
}

func (s *Sampler) Shoot(camera Camera) (imag *Image) {
	imag = &Image{}
	imag.Height = camera.Vres
	imag.Width = camera.Hres
	imag.Pixels = make([][]Colour, camera.Vres)

	type Job struct{ x, y int }
	type Result struct {
		x, y int
		col  Colour
	}

	inCh := make(chan Job, camera.Hres*camera.Vres)
	outCh := make(chan Result, 10)

	for i := 0; i < runtime.NumCPU(); i++ {
		go func() {
			for {
				job, more := <-inCh

				if !more {
					break
				}

				down := camera.Down.Scale(float64(2*job.x-camera.Vres) / 2)
				right := camera.Right.Scale(float64(2*job.y-camera.Hres) / 2)
				point := camera.Centre.Add(down).Add(right)
				ray := Ray{camera.Origin, point.Normalize()}
				outCh <- Result{job.x, job.y, s.Sample(ray)}
			}
		}()
	}

	for i := 0; i < camera.Vres; i++ {
		imag.Pixels[i] = make([]Colour, camera.Hres)

		for j := 0; j < camera.Hres; j++ {
			inCh <- Job{i, j}
		}
	}
	close(inCh)

	nPixels := 0
	for {
		res := <-outCh
		imag.Pixels[res.x][res.y] = res.col
		nPixels++
		if nPixels == camera.Vres*camera.Hres {
			break
		}
	}

	return imag
}

type Camera struct {
	Origin, Centre, Down, Right Vector
	Hres, Vres                  int
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

func (h reflectiveSphereHit) Next() (*Ray, *Colour) {
	return nil, nil
}

func (sphere *ReflectiveSphere) Intersect(ray Ray) Hit {
	var ret reflectiveSphereHit
	projCentre := ray.Project(sphere.Centre)

	if !ray.InView(projCentre) {
		return nil
	}

	a := projCentre.Sub(sphere.Centre).Length()
	// b := math.Sqrt(sphere.Radius*sphere.Radius - a*a)
	ret.distance = projCentre.Sub(ray.Origin).Length()

	if a >= sphere.Radius {
		return nil
	}

	return &ret
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

	distance := -offset / direction

	if distance < 0 {
		// ray moved away from the hyperplane - not hit
		return nil
	}

	return &hcbHit{ray, b, distance}
}

func (b *HyperCheckerboard) Origin() Vector {
	return b.Normal.Origin
}

type hcbHit struct {
	ray      Ray
	board    *HyperCheckerboard
	distance float64
}

func (h *hcbHit) Distance() float64 {
	return h.distance
}

func (h *hcbHit) Next() (ray *Ray, colour *Colour) {
	intercept := h.ray.Follow(h.distance)
	t := make([]float64, len(h.board.Axes))

	for i := 0; i < len(t); i++ {
		axisray := Ray{h.board.Origin(), h.board.Axes[i].Normalize()}
		t[i] = axisray.RelativeLength(intercept)
		t[i] = t[i] / h.board.Axes[i].Length()
	}

	sum := 0
	for i := 0; i < len(t); i++ {
		sum += int(math.Floor(t[i]))
	}
	sign := sum % 2

	if sign == 1 {
		colour = &Black
	} else {
		colour = &White
	}

	return
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())

	N := 3
	Hres := 1000
	Vres := 1000

	origin := V(make([]float64, N)...)
	up := E(N, 0)

	sphere := &ReflectiveSphere{}
	sphere.Centre = up
	sphere.Radius = 1

	floor := &HyperCheckerboard{}
	floor.Normal = Ray{origin, up.Normalize()}
	floor.Axes = make([]Vector, N-1)
	for i := 0; i < N-1; i++ {
		floor.Axes[i] = E(N, i+1)
	}

	ceiling := &HyperCheckerboard{}
	ceiling.Normal = Ray{up.Scale(2), up.Normalize()}
	ceiling.Axes = floor.Axes

	scene := &Scene{}
	scene.Bodies = []Body{floor, ceiling}

	camera := Camera{}
	corigin := make([]float64, N)
	corigin[0] = 1
	corigin[1] = -2
	camera.Origin = Vector(corigin)

	camera.Centre = E(N, 1)
	camera.Down = origin.Sub(up).Scale(1 / float64(Vres))
	camera.Right = E(N, 2).Scale(1 / float64(Hres))
	camera.Hres = Hres
	camera.Vres = Vres

	sampler := &Sampler{}
	sampler.Root = scene
	sampler.MaxBounces = 10
	sampler.FirstBatch = 10
	sampler.Target = .2

	file, _ := os.Create("test.png")
	png.Encode(file, sampler.Shoot(camera).ToNRGBA())
	file.Close()
}
