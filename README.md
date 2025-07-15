# lumaray

scuffed rust raytracer

* lumaray: raytracing kernel
* qndview: "quick and dirty" mesh viewer app using ggez

* extremely fast avx512 bvh/triangle intersection, using a "two-volume" bbox representation where two bounding boxes are packed into a simd register
* a broken mesh chunking algo, needs to be replaced with a mesh walk
* mesh instancting and transforms
* multithreading obviously
* idk

## Building:

* avx2: `cargo build`
* avx512: `cargo build --features avx512`
* no-simd: `cargo build --features no-simd`
