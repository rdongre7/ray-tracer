# Assignment 5 extension: Accelerating ray intersection

In the CS4621 extension to the ray tracing assignment, we ask you to implement a ray-tracing acceleration structure for your Python ray tracer. 

## Requirements

* Add a class `Accel` to the ray package, with an initializer that accepts a list of `Mesh` objects and an `intersect` method with the same interface as `Mesh.intersect`.
* Render images identical to A5 when meshes are wrapped in `Accel` objects.
* Render scenes containing large numbers of triangles efficiently.

A working definition of “efficiently” is that you should be able to render the `two_bunnies.py` test scene (10k triangles) in about 2-5 times as much time as the `mesh_cube.py` scene (12 triangles).

For instance, in an A5 input file you could make a scene with two meshes this way:

~~~
def read_mesh(fname, material):
    i, p, n, t = read_obj(open(fname))
    return Mesh(i, p, n, t, material)
scene = Scene([
   read_mesh(“mesh1.obj”, mat1),
   read_mesh(“mesh2.obj”, mat1)
])
~~~

and with the same definitions once your acceleration structure is done you will be able to instead write

~~~
scene = Scene([
   Accel([
      read_mesh(“mesh1.obj”, mat1),
      read_mesh(“mesh2.obj”, mat1)
   ])
])
~~~

and render the same image faster.  The acceleration structure does not need to handle spheres; they can just be added to the scene separately from the `Accel` object.  Nothing stops you from putting multiple `Accel` objects in the scene but there’s no need; you can put all the meshes into one acceleration structure.

We have provided a much larger scene (over half a million triangles) called `buddhas.py` which our solution renders at the default 256 pixel width in about 5 minutes.

You can likely achieve this performance using a bounding box tree (as we did), a k-D tree, a sphere tree, or a regular grid.  You are not required to use any particular algorithm, but in the lecture and in this writeup we discuss axis-aligned bounding box (AABB) trees in detail.

The code for this project does not need to be long (our solution adds 125 lines to ray.py for the acceleration structure), but it requires some thought in the design and it can be tricky to debug.


## Implementation notes

Your implementation of the `Accel` class will build a data structure in the initializer that organizes all the triangles in all the meshes it is given, and then search that data structure in the `intersect` method.  You can do this with a binary tree built from Python objects (though you could also use array-based storage schemes), each of which has either two children or a list of triangles.  It's usual to keep these two cases separate, with triangles contained only at the leaf nodes.

If you use an AABB tree, then each node will store an AABB (just a min and max point in 3D), and the intersection algorithm is very simple:
* Intersect ray and bounding box.  If it misses, return no hit.
* Otherwise, intersect ray with contents.  For internal nodes the contents are the two child nodes; for leaf nodes the contents is a list of triangles.  In both cases you find and return the hit associated with the smallest _t_ value.

So for intersection there are two key pieces of code: ray/box intersection and ray intersection with a list of triangles.  Of course the latter is something you already implemented for A5.

Before you can intersect rays with the tree, you have to build the tree.  Building an AABB tree for a list of triangles is a recursive process with just a few steps:
* If there are few enough triangles, create a leaf node and return it.
* Otherwise, split the list of triangles in two, build the two subtrees recursively, then create an internal node with those two children and return it.

To do this you need a way to split the list in two.  The canonical choice is to sort the triangles along the longest axis of the bounding box, then break the list into two contiguous pieces.  The lecture discusses some ways of deciding where to divide the list; the median split is probably simplest but an equal-surface-area split usually performs better.

There are a number of design choices in how you store the actual triangle data.  The leaf nodes can be little self-contained meshes with their own lists of indices, positions, normals, and uvs.  Or you can leave the data in the original meshes and have your leaf nodes contain lists of (mesh index, triangle index) pairs to reference triangles in those meshes.  Or you can copy the index and position data into one big array and reference it from the leaf nodes using lists of indices into that array.  Or (our favorite) you can sort the big array in place so that each leaf node doesn't need a list of indices, just a start and end index in the big array.

You will want to test with small meshes and small numbers of triangles in leaf nodes, so you can print out your data structures in debugging.  But once your tree works, an important tuning parameter to explore is the threshold for how many triangles are allowed in a leaf node.  We found the best number to be pretty big, around 100, but your own optimum value will depend on the relative speed of the different parts of your code.  You'll find a few meshes of varying size in the `models/` directory.

One problem that often comes up with AABBs is computing the bounding box for axis-aligned sets of triangles that end up with zero thickness along one axis.  This might happen more often in testing than in practice (if you use very small leaf nodes and simple meshes like the cube), but it comes up in real scenes that have large axis-aligned walls and floors.  It is good to expand your boxes by a small epsilon on all sides to make sure that the contents are unambiguously inside the box, even in the face of finite precision.  We used an epsilon of `1e-6`.


## Demo scene

Make a scene to demo your ray tracer.  It can be an extension of one you used for the A4 or A5 creative part.  But you have a much larger triangle budget now so you can probably make something more interesting!  Feel free to use meshes from wherever you like, but cite your sources.  You can follow the example of `buddhas.py` for how to position objects by applying transformations to them.  Don't forget you can render at super low resolution while iterating on your scene.


## Handing in

Hand in a Zip file containing `ray.py` (it is probably self-contained but if you need more .py files feel free), your new scene, output images for the test scenes and your new scene, a README file that explains how you designed your software and how you built your scene.
