package main

import (
	"flag"
	"image"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	"github.com/google/gxui"
	"github.com/google/gxui/drivers/gl"
	"github.com/google/gxui/themes/dark"
)

const N = 41

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

type Ray struct {
	Origin    Vector
	Direction UnitVector
}

// m-th standard basis vector in N dimensions
func E(N, m int) (res Vector) {
	res[m] = 1
	return
}

func (v Vector) Add(w *Vector) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] + w[i]
	}
	return
}

func (v *Vector) Sub(w *Vector) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] - w[i]
	}
	return
}

func (v *Vector) Dot(w *Vector) (res float64) {
	for i := 0; i < N; i++ {
		res += v[i] * w[i]
	}
	return
}

func (v *Vector) Length() float64 {
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

func (v *UnitVector) Dot(w *Vector) (res float64) {
	for i := 0; i < N; i++ {
		res += v[i] * w[i]
	}
	return
}

func (v UnitVector) Scale(scalar float64) (ret Vector) {
	for i := 0; i < N; i++ {
		ret[i] = v[i] * scalar
	}
	return
}

// Returns the length of the vector projected onto the line associated with the
// ray relative to the origin of the ray.
func (r *Ray) RelativeLength(v *Vector) float64 {
	moved := v.Sub(&r.Origin)
	return r.Direction.Dot(&moved)
}

func (r *Ray) InView(v *Vector) bool {
	return r.RelativeLength(v) >= 0
}

// Returns the vector v projected onto the line associated with the ray
func (r *Ray) Project(v *Vector) Vector {
	return r.Direction.Scale(r.RelativeLength(v)).Add(&r.Origin)
}

func (r *Ray) Follow(distance float64) Vector {
	return r.Direction.Scale(distance).Add(&r.Origin)
}

func (incoming UnitVector) Reflect(normal UnitVector) (outgoing UnitVector) {
	nv := Vector(normal)
	iv := Vector(incoming)
	v2 := nv.Scale(2 * iv.Dot(&nv))
	outgoing = iv.Sub(&v2).Normalize()
	return
}

type Hit struct {
	distance  float64
	body      Body
	ray       Ray
	intercept Vector
}

type Body interface {
	Intersect(Ray, *Hit) bool

	// If hit, computes what happens next.  Returns a
	// triplet (lambda, ray, colour) which should be understood as a
	// convex combination lambda | ray > + (1 - lambda) | colour >,
	// where colour is the absorped part and ray is the reflected/refracted/etc
	// ray.
	Next(*Hit, *rand.Rand) (float64, Ray, Colour)
}

type Scene struct {
	Bodies []Body
}

// Scene currently simply checks all bodies for a hit.  For a bigger scenes,
// we might want to add bounding boxes to Body's and add a tree.
func (s *Scene) Intersect(ray Ray, minDistHit *Hit) (ok bool) {
	var hit Hit
	minDist := math.Inf(1)
	for _, body := range s.Bodies {
		var ok2 bool
		switch body2 := body.(type) {
		case *Scene:
			ok2 = body2.Intersect(ray, &hit)
		case *HyperCheckerboard:
			ok2 = body2.Intersect(ray, &hit)
		case *ReflectiveSphere:
			ok2 = body2.Intersect(ray, &hit)
		default:
			panic("eh?")
		}
		if !ok2 {
			continue
		}
		if hit.distance < minDist && hit.distance > eps {
			minDist = hit.distance
			*minDistHit = hit
			ok = true
		}
	}
	return
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

func (s *Sampler) SampleOne(ray Ray, dx, dy Vector, rnd *rand.Rand) (ret Colour) {
	dx = dx.Scale(rnd.Float64())
	dy = dy.Scale(rnd.Float64())
	var factor float64 = 1.0
	var hit Hit
	var colour Colour
	var lambda float64

	ray.Direction = Vector(ray.Direction).Add(&dx).Add(&dy).Normalize()

	body := s.Root
	for i := 0; i < s.MaxBounces; i++ {
		var ok bool
		switch body2 := body.(type) {
		case *Scene:
			ok = body2.Intersect(ray, &hit)
		case *HyperCheckerboard:
			ok = body2.Intersect(ray, &hit)
		case *ReflectiveSphere:
			ok = body2.Intersect(ray, &hit)
		default:
			panic("eh?")
		}
		if !ok {
			return
		}
		switch body2 := hit.body.(type) {
		case *Scene:
			lambda, ray, colour = body2.Next(&hit, rnd)
		case *HyperCheckerboard:
			lambda, ray, colour = body2.Next(&hit, rnd)
		case *ReflectiveSphere:
			lambda, ray, colour = body2.Next(&hit, rnd)
		default:
			panic("eh?")
		}
		if lambda == 0 {
			ret = ret.Add(colour.Scale(factor))
			return
		}
		ret = ret.Add(colour.Scale(factor * (1 - lambda)))
		factor *= lambda
		if factor < s.Target {
			return
		}
	}
	log.Print("MaxBounces hit :(")
	return Black
}

func (s *Sampler) SampleBatch(ray Ray, size int, dx, dy Vector, rnd *rand.Rand) (c Colour) {
	for i := 0; i < size; i++ {
		c = c.Add(s.SampleOne(ray, dx, dy, rnd))
	}
	c = c.Scale(1 / float64(size))
	return
}

func (s *Sampler) Sample(ray Ray, dx, dy Vector, rnd *rand.Rand) Colour {
	var old, new Colour

	size := s.FirstBatch

	old = s.SampleBatch(ray, size, dx, dy, rnd)
	new = s.SampleBatch(ray, size, dx, dy, rnd)

	for {
		if old.Sub(new).SupNorm() < s.Target {
			return old.Add(new).Scale(.5)
		}

		old = old.Add(new).Scale(.5)
		size = 2 * size
		new = s.SampleBatch(ray, size, dx, dy, rnd)
	}
}

func (s *Sampler) Shoot(camera Camera, imag *image.RGBA) {
	type Job struct{ x int }
	type Result struct {
		x   int
		col []Colour
	}

	inCh := make(chan Job, camera.Hres*camera.Vres)
	outCh := make(chan Result, 10)

	for i := 0; i < runtime.NumCPU(); i++ {
		go func() {
			rnd := rand.New(rand.NewSource(rand.Int63()))
			for {
				job, more := <-inCh

				if !more {
					break
				}

				cols := make([]Colour, camera.Hres)

				for y := 0; y < camera.Hres; y++ {
					down := camera.Down.Scale(float64(2*job.x-camera.Vres) / 2)
					right := camera.Right.Scale(float64(2*y-camera.Hres) / 2)
					point := camera.Centre.Add(&down).Add(&right)
					ray := Ray{camera.Origin, point.Normalize()}
					cols[y] = s.Sample(ray, camera.Right, camera.Down, rnd)
				}

				outCh <- Result{job.x, cols}
			}
		}()
	}

	for i := 0; i < camera.Vres; i++ {
		inCh <- Job{i}
	}
	close(inCh)

	nPixels := 0
	for {
		res := <-outCh
		for y := 0; y < camera.Hres; y++ {
			imag.Set(y, res.x, res.col[y])
		}
		nPixels += camera.Hres
		log.Printf("%d pixels", nPixels)
		if nPixels == camera.Vres*camera.Hres {
			break
		}
	}
}

type Camera struct {
	Origin, Centre, Down, Right Vector
	Hres, Vres                  int
}

type ReflectiveSphere struct {
	Centre Vector
	Radius float64
}

func (scene *Scene) Next(hit *Hit, rnd *rand.Rand) (lambda float64, ray Ray, colour Colour) {
	panic("Scene.Next() should not have been called")
}

func (sphere *ReflectiveSphere) Next(hit *Hit, rnd *rand.Rand) (lambda float64, ray Ray, colour Colour) {
	lambda = 1
	normal := hit.intercept.Sub(&sphere.Centre).Normalize()
	direction := hit.ray.Direction.Reflect(normal)

	ray = Ray{hit.intercept, direction}
	return
}

func (sphere *ReflectiveSphere) Intersect(ray Ray, hit *Hit) bool {
	hit.ray = ray
	hit.body = sphere
	projCentre := ray.Project(&sphere.Centre)

	if !ray.InView(&projCentre) {
		return false
	}

	aVec := projCentre.Sub(&sphere.Centre)
	a := aVec.Length()

	if a >= sphere.Radius {
		return false
	}

	b := math.Sqrt(sphere.Radius*sphere.Radius - a*a)
	d := projCentre.Sub(&ray.Origin)
	hit.distance = d.Length() - b
	hit.intercept = hit.ray.Follow(hit.distance)

	return true
}

type HyperCheckerboard struct {
	Normal Ray
	Axes   []Vector

	AxisRays    []Ray
	AxisLengths []float64
}

func NewHyperCheckboard(normal Ray, axes []Vector) *HyperCheckerboard {
	ret := &HyperCheckerboard{}
	ret.Normal = normal
	ret.Axes = axes
	ret.AxisRays = make([]Ray, len(axes))
	ret.AxisLengths = make([]float64, len(axes))
	for i := 0; i < len(axes); i++ {
		ret.AxisRays[i] = Ray{normal.Origin, axes[i].Normalize()}
		ret.AxisLengths[i] = axes[i].Length()
	}
	return ret
}

func (b *HyperCheckerboard) Intersect(ray Ray, hit *Hit) bool {
	hit.body = b
	hit.ray = ray
	offset := b.Normal.RelativeLength(&ray.Origin)
	direction := b.Normal.Direction.Dot((*Vector)(&ray.Direction))

	if math.Abs(direction) <= eps {
		// ray is parallel to hyperplane - not hit
		return false
	}

	hit.distance = -offset / direction

	if hit.distance < 0 {
		// ray moved away from the hyperplane - not hit
		return false
	}

	hit.intercept = ray.Follow(hit.distance)
	if hit.intercept.Length() > 15 {
		return false
	}

	return true
}

func (b *HyperCheckerboard) Origin() Vector {
	return b.Normal.Origin
}

type hcbHit struct {
	ray       Ray
	board     *HyperCheckerboard
	distance  float64
	intercept Vector
}

func (h *hcbHit) Distance() float64 {
	return h.distance
}

func (b *HyperCheckerboard) Next(hit *Hit, rnd *rand.Rand) (lambda float64, ray Ray, colour Colour) {
	lambda = 0.2
	sum := 0
	for i := 0; i < len(b.Axes); i++ {
		t := b.AxisRays[i].RelativeLength(&hit.intercept) / b.AxisLengths[i]
		sum += int(math.Abs(math.Floor(t)))
	}
	sign := sum % 2

	if sign == 1 {
		colour = Black
	} else {
		colour = White
	}

	ray.Origin = hit.intercept
	ray.Direction = hit.ray.Direction.Reflect(b.Normal.Direction)
	return
}

func main() {
	var cpuprofile = flag.String("cpuprofile", "", "write cpu profile")
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

	gl.StartDriver(glMain)
}

func glMain(driver gxui.Driver) {
	Hres := 750
	Vres := 750

	e0 := E(N, 0)

	sphere := &ReflectiveSphere{}

	sphere.Radius = 0.90

	floorAxes := make([]Vector, N-1)
	for i := 0; i < N-1; i++ {
		floorAxes[i] = E(N, i+1)
	}
	floor := NewHyperCheckboard(Ray{e0, e0.Scale(-1).Normalize()}, floorAxes)
	ceiling := NewHyperCheckboard(Ray{e0.Scale(-1), e0.Normalize()}, floorAxes)

	scene := &Scene{}
	scene.Bodies = []Body{floor, ceiling, sphere}

	camera := Camera{}
	camera.Origin[1] = -2
	camera.Origin[2] = -2

	camera.Centre = camera.Origin.Scale(-.4)

	// camera.Down = [...]float64{-1, 0, 0, 0, 0}
	// camera.Down = [...]float64{-1, 0, 0, 0, 1}
	camera.Down[0] = -1
	camera.Right[1] = 1
	camera.Right[2] = -1

	for i := 0; i < (N-3)/2; i++ {
		camera.Right[3+2*i] = math.Pow(2, -float64(i)-1)
		camera.Down[4+2*i] = math.Pow(2, -float64(i)-1)
	}

	// camera.Down = [...]float64{-1, 0, 0, 0, 0.5, 0, .25, 0, .125}
	// camera.Down = [...]float64{-1, 0, 0, 0, 0.5, 0, 0}
	// camera.Right = [...]float64{0, 1, -1, 0, 0}
	// camera.Right = [...]float64{0, 1, -1, .5, 0, .25, 0, .125, 0}
	// camera.Right = [...]float64{0, 1, -1, .5, 0, 0, 0}

	camera.Right = camera.Right.Normalize().Scale(1 / float64(Hres))
	camera.Down = camera.Down.Normalize().Scale(1 / float64(Vres))
	camera.Hres = Hres
	camera.Vres = Vres

	sampler := &Sampler{}
	sampler.Root = scene
	sampler.MaxBounces = 20
	sampler.FirstBatch = 10

	// UI stuff
	m := image.NewRGBA(image.Rect(0, 0, Hres, Vres))

	theme := dark.CreateTheme(driver)
	img := theme.CreateImage()
	window := theme.CreateWindow(Hres, Vres, "ndpt")
	window.AddChild(img)
	window.OnClose(driver.Terminate)

	// render render
	go func() {
		sampler.Target = .9
		sampler.Shoot(camera, m)
		sampler.Target = .05
		sampler.Shoot(camera, m)
		sampler.Target = .01
		sampler.Shoot(camera, m)
		sampler.Target = .005
		sampler.Shoot(camera, m)
	}()

	go func() {
		for {
			time.Sleep(100 * time.Millisecond)
			driver.Call(func() {
				texture := driver.CreateTexture(m, 1.0)
				img.SetTexture(texture)
			})
		}
	}()
}
