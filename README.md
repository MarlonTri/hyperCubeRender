## Hypercube embeddings

Cubes of various dimensions embedded into 2D space using the [Isomap](https://en.wikipedia.org/wiki/Isomap) algorithm.

### Inspiration:

I recently got 4 paint pigments come in the mail, all unique colors. The solitary colors are a bit ugly, but I bet there is some combination of the pigments that would look great. If I only had 2 pigment colors, I could just paint a gradient as a line, going from 100% pigment A to 100% pigment B. For 3 colors, all combinations could be represented as a [ternary plot](https://en.wikipedia.org/wiki/Ternary_plot).

Realistically, a "tetrahedron plot" could represent all 4 color pigment combinations in 3d space. I got sidetracked and decided to look into cube representations instead. I chose to try and represent all pigment combinations as a N-dimensional hypercube embedded into 2D space using the Isomap algorithm and other similar nonlinear dimensionality reduction methods. So I implemented it!

### What's the rendering algorithm?
 1. I create a meshgrid of the N-dimensional cube. If the meshgrid has resolution `R`, there will be `R**N` total points.
 1. I apply the Isomap algorithm to that meshgrid, recieving an embedding function into 2D space.
 1. At this point, the embedding is rough and has artifacting. Also it's incredibly slow.
 2. Therefor, I the `scikit` function `RegularGridInterpolator` to interpolate outputs from that initial meshgrid embedding.
 3. Finally, I generate N unique colors, assigning one to each dimension.
 4. I randomly sample points in the N-dimensional cube.
    1. The distribution is biased towards edges to give more definition to the cube.
    2. The point is assigned a color: the linear combination of the N dimensional colors based off it's location
    3. The point is assigned a 2D coordinate based of the Isomap embedding
 5. After enough samples, the noise becomes an image!


### Cool takeaways:
 - The amount of edges connected to each vertex is equal to the dimension.
   - A square has two edges connected to vertex.
   - A cube has two edges connected to vertex
 - This tool is not really great for my original usecase of finding a good pigment mixture.
 - Sometimes function interpolations are great for speeding up compute without loosing fidelity.


<!-- ======================= -->
<!-- 3D RENDERS -->
<!-- ======================= -->
<h2>3D cubes</h2>

### Simply a regular ol' 3D cube embedded into 2D space.
<!-- Always-visible previews (first 3) -->
<div style="display:flex; flex-wrap:wrap; gap:8px;">
  <img src="renders/3D_07rez_render.png" width="300"/>
  <img src="renders/3D_11rez_render.png" width="300"/>
  <img src="renders/3D_13rez_render.png" width="300"/>
</div>

<!-- Foldable remainder -->
<details>
  <summary><strong>More 3D renders</strong></summary>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
    <img src="renders/3D_03rez_render.png" width="300"/>
    <img src="renders/3D_04rez_render.png" width="300"/>
    <img src="renders/3D_05rez_render.png" width="300"/>
    <img src="renders/3D_06rez_render.png" width="300"/>
    <img src="renders/3D_08rez_render.png" width="300"/>
    <img src="renders/3D_09rez_render.png" width="300"/>
    <img src="renders/3D_10rez_render.png" width="300"/>
    <img src="renders/3D_12rez_render.png" width="300"/>
    <img src="renders/3D_14rez_render.png" width="300"/>
  </div>
</details>


<!-- ======================= -->
<!-- 4D RENDERS -->
<!-- ======================= -->
<h2>4D cubes</h2>

### A 4D hypercube embedded into 2D space - it's essentially a pair of 3D cubes glued together, just as a cube is two squares glued together.

<!-- Always-visible previews (first 3) -->
<div style="display:flex; flex-wrap:wrap; gap:8px;">
  <img src="renders/4D_05rez_render.png" width="300"/>
  <img src="renders/4D_06rez_render.png" width="300"/>
  <img src="renders/4D_09rez_render.png" width="300"/>
</div>

<!-- Foldable remainder -->
<details>
  <summary><strong>More 4D renders</strong></summary>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
    <img src="renders/4D_03rez_render.png" width="300"/>
    <img src="renders/4D_04rez_render.png" width="300"/>
    <img src="renders/4D_07rez_render.png" width="300"/>
    <img src="renders/4D_08rez_render.png" width="300"/>
  </div>
</details>


<!-- ======================= -->
<!-- 5D RENDERS -->
<!-- ======================= -->
<h2>5D cubes</h2>

### A 5-dimensional cube embedded into 2D space. Notice each vertex has 5 edges connected.
<!-- Always-visible previews (first 3) -->
<div style="display:flex; flex-wrap:wrap; gap:8px;">
  <img src="renders/5D_03rez_render.png" width="300"/>
  <img src="renders/5D_05rez_render.png" width="300"/>
  <img src="renders/5D_06rez_render.png" width="300"/>
</div>

<!-- Foldable remainder -->
<details>
  <summary><strong>More 5D renders</strong></summary>
  <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
    <img src="renders/5D_04rez_render.png" width="300"/>
    <img src="renders/5D_07rez_render.png" width="300"/>
  </div>
</details>

<h2>6D cubes</h2>

### 6D cubes are essentially incoherent to us humans
<!-- Always-visible previews (first 3) -->
<div style="display:flex; flex-wrap:wrap; gap:8px;">
  <img src="renders/6D_03rez_render.png" width="300"/>
  <img src="renders/6D_04rez_render.png" width="300"/>
  <img src="renders/6D_05rez_render.png" width="300"/>
</div>
