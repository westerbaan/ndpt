package main

import (
	"flag"
	"image"
	"image/png"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
)

const N = 5

type Vector [N]float64
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

// m-th standard basis vector in N dimensions
func E(N, m int) (res Vector) {
	res[m] = 1
	return
}

func (v Vector) Add(w Vector) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] + w[i]
	}
	return
}

func (v Vector) Sub(w Vector) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] - w[i]
	}
	return
}

func (v Vector) Dot(w Vector) (res float64) {
	for i := 0; i < N; i++ {
		res += v[i] * w[i]
	}
	return
}

func (v Vector) Length() float64 {
	return math.Sqrt(v.Dot(v))
}

func (v Vector) Scale(scalar float64) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] * scalar
	}
	return
}

func (v Vector) Normalize() UnitVector {
	return UnitVector(v.Scale(1.0 / v.Length()))
}

// Returns whether the vector is almost zero, see eps.
func (v Vector) IsZero() bool {
	return v.Length() < eps
}

func (v UnitVector) Dot(w Vector) (res float64) {
	for i := 0; i < N; i++ {
		res += v[i] * w[i]
	}
	return
}

func (v UnitVector) Scale(scalar float64) Vector {
	return Vector(v).Scale(scalar)
}

// Returns the length of the vector projected onto the line associated with the
// ray relative to the origin of the ray.
func (r *Ray) RelativeLength(v Vector) float64 {
	return v.Sub(r.Origin).Dot(Vector(r.Direction))
}

func (r *Ray) InView(v Vector) bool {
	return r.RelativeLength(v) >= 0
}

// Returns the vector v projected onto the line associated with the ray
func (r *Ray) Project(v Vector) Vector {
	return Vector(r.Direction).Scale(r.RelativeLength(v)).Add(r.Origin)
}

func (r *Ray) Follow(distance float64) Vector {
	return r.Origin.Add(r.Direction.Scale(distance))
}

func (incoming UnitVector) Reflect(normal UnitVector) (outgoing UnitVector) {
	nv := Vector(normal)
	iv := Vector(incoming)
	outgoing = iv.Sub(nv.Scale(2 * iv.Dot(nv))).Normalize()
	return
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
		if dist < minDist && dist > eps {
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

func (s *Sampler) SampleOne(ray Ray, dx, dy Vector) Colour {
	dx = dx.Scale(rand.Float64())
	dy = dy.Scale(rand.Float64())

	ray.Direction = Vector(ray.Direction).Add(dx).Add(dy).Normalize()

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

func (s *Sampler) SampleBatch(ray Ray, size int, dx, dy Vector) (c Colour) {
	for i := 0; i < size; i++ {
		c = c.Add(s.SampleOne(ray, dx, dy))
	}
	c = c.Scale(1 / float64(size))
	return
}

func (s *Sampler) Sample(ray Ray, dx, dy Vector) Colour {
	var old, new Colour

	size := s.FirstBatch

	old = s.SampleBatch(ray, size, dx, dy)
	new = s.SampleBatch(ray, size, dx, dy)

	for {
		if old.Sub(new).SupNorm() < s.Target {
			return old.Add(new).Scale(.5)
		}

		old = old.Add(new).Scale(.5)
		size = 2 * size
		new = s.SampleBatch(ray, size, dx, dy)
	}
}

func (s *Sampler) Shoot(camera Camera) (imag *Image) {
	imag = &Image{}
	imag.Height = camera.Vres
	imag.Width = camera.Hres
	imag.Pixels = make([][]Colour, camera.Vres)

	type Job struct{ x int }
	type Result struct {
		x   int
		col []Colour
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

				cols := make([]Colour, camera.Hres)

				for y := 0; y < camera.Hres; y++ {
					down := camera.Down.Scale(float64(2*job.x-camera.Vres) / 2)
					right := camera.Right.Scale(float64(2*y-camera.Hres) / 2)
					point := camera.Centre.Add(down).Add(right)
					ray := Ray{camera.Origin, point.Normalize()}
					cols[y] = s.Sample(ray, camera.Right, camera.Down)
				}

				outCh <- Result{job.x, cols}
			}
		}()
	}

	for i := 0; i < camera.Vres; i++ {
		imag.Pixels[i] = make([]Colour, camera.Hres)
		inCh <- Job{i}
	}
	close(inCh)

	nPixels := 0
	for {
		res := <-outCh
		for y := 0; y < camera.Hres; y++ {
			imag.Pixels[res.x][y] = res.col[y]
		}
		nPixels += camera.Hres
		log.Printf("%d pixels", nPixels)
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
	ray      Ray
	Sphere   *ReflectiveSphere
	distance float64
}

func (h reflectiveSphereHit) Distance() float64 {
	return h.distance
}

func (h reflectiveSphereHit) Next() (ray *Ray, colour *Colour) {
	intercept := h.ray.Follow(h.distance)
	normal := intercept.Sub(h.Sphere.Centre).Normalize()
	direction := h.ray.Direction.Reflect(normal)

	ray = &Ray{intercept, direction}

	//colour = &White
	return
}

func (sphere *ReflectiveSphere) Intersect(ray Ray) Hit {
	var ret reflectiveSphereHit
	ret.ray = ray
	ret.Sphere = sphere
	projCentre := ray.Project(sphere.Centre)

	if !ray.InView(projCentre) {
		return nil
	}

	a := projCentre.Sub(sphere.Centre).Length()
	b := math.Sqrt(sphere.Radius*sphere.Radius - a*a)
	ret.distance = projCentre.Sub(ray.Origin).Length() - b

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

	if ray.Follow(distance).Length() > 15 {
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
		sum += int(math.Abs(math.Floor(t[i])))
	}
	sign := sum % 2

	if rand.Intn(5) != 0 {
		if sign == 1 {
			colour = &Black
		} else {
			colour = &White
		}
		return
	}

	ray = &Ray{
		Origin:    intercept,
		Direction: h.ray.Direction.Reflect(h.board.Normal.Direction),
	}

	return
}

func main() {
	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile")
	var memprofile = flag.String("memprofile", "", "write mem profile")
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	Hres := 500
	Vres := 500

	var origin Vector
	up := E(N, 0)

	sphere := &ReflectiveSphere{}

	sphere.Centre = [...]float64{1, .2, .2, .2, .2}
	sphere.Radius = .9

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
	scene.Bodies = []Body{floor, ceiling, sphere}

	camera := Camera{}
	camera.Origin = sphere.Centre.Add(E(N, 1).Scale(-3))

	ccentre := make([]float64, N)
	ccentre[0] = 0
	ccentre[1] = 1
	ccentre[3] = .06
	ccentre[4] = .07
	camera.Centre = [...]float64{0, 1, 0, .06, .07}
	camera.Down = origin.Sub(up).Scale(1 / float64(Vres))
	camera.Right = E(N, 2).Scale(1 / float64(Hres))
	camera.Hres = Hres
	camera.Vres = Vres

	sampler := &Sampler{}
	sampler.Root = scene
	sampler.MaxBounces = 20
	sampler.FirstBatch = 30
	sampler.Target = .05

	file, _ := os.Create("test.png")
	png.Encode(file, sampler.Shoot(camera).ToNRGBA())
	file.Close()

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		runtime.GC() // get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
		f.Close()
	}
}
