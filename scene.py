from utils import *
from ray import *
from cli import render

rngB = np.random.default_rng(0)


def rand_color():
    return Material(rngB.random(3), 0.6)


red = Material(vec([0.7019, 0.106, 0.106]), k_s=0.3, p=30, k_m=0.3)
gray = Material(vec([0.1333, 0.1333, 0.1333]), k_m=0.5)
white = Material(vec([1, 1, 1]), k_m=0.4)
tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]))


tanB = Material(vec([0.7, 0.7, 0.4]), 0.6)
grayB = Material(vec([0.2, 0.2, 0.2]))

vs_list = 0.5 * read_obj_triangles(open("models/cube.obj"))
vs_list2 = 0.25 * (read_obj_triangles(open("models/cube.obj"))) + 2
vs_list3 = 0.2 * (read_obj_triangles(open("models/cube.obj"))) + 8

# ====================================================s==========================
# mesh_cube_tex.py


# # For step 3, uncomment line 7 (i.e. use the faces texture) to match the reference image
# # This should also give the same textured cube as cube_tex
# tan = Material(load_image('textures/pips.png') * [[0.7, 0.7, 0.4]])
# # tan = Material(load_image('textures/faces.png'))
# gray = Material(vec([0.2, 0.2, 0.2]))

# # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
# (i, p, n, t) = read_obj(open("models/cube.obj"))

# scene = Scene([
#     # Make a big sphere for the floor
#     Sphere(vec([0, -40, 0]), 39.5, gray),
# ] + [
#     Mesh(i, 0.5*p, None, t, tan),
# ])

# lights = [
#     PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#     AmbientLight(0.1),
# ]

# camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16/9)

# # ==============================================================================
# # sphere_globe

# maptex = Material(load_image('textures/map.png'))
# gray = Material(vec([0.2, 0.2, 0.2]))

# scene = Scene([
#     Sphere(vec([0, 0, 0]), 0.5, maptex),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
# ])

# lights = [
#     PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#     AmbientLight(0.1),
# ]

# camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=17, aspect=16/9)


# # ==============================================================================

# # three_spheres_texture

# tan = Material(vec([0.4, 0.4, 0.2]), k_s=load_image('textures/specular-map.png'),
#                p=load_image('textures/mirror-texture.png'), k_m=0.3)
# blue = Material(vec([0.2, 0.2, 0.5]), k_a=load_image(
#     'textures/map.png'), k_m=0.5)
# gray = Material(vec([0.2, 0.2, 0.2]), k_m=load_image(
#     'textures/mirror-texture-metal-foil.png'))

# scene = Scene([
#     Sphere(vec([-0.7, 0, 0]), 0.5, tan),
#     Sphere(vec([0.7, 0, 0]), 0.5, blue),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
# ])

# lights = [
#     PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#     AmbientLight(0.1),
# ]

# camera = Camera(vec([3, 1.2, 5]), target=vec(
#     [0, -0.4, 0]), vfov=24, aspect=16/9)


# # ==============================================================================
# # two globe

# maptex = Material(load_image('textures/map.png'))
# gray = Material(vec([0.2, 0.2, 0.2]))

# # Read the triangle mesh for a triangulated unit sphere
# (i, p, n, t) = read_obj(open("models/uvsphere_smooth.obj"))

# scene = Scene([
#     Sphere(vec([-0.7, 0, 0]), 0.5, maptex),
#     Mesh(i, 0.5*p + [[0.7, 0.0, 0.0]], n, t, maptex),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
# ])

# lights = [
#     PointLight(vec([12, 10, 8]), vec([300, 300, 300])),
#     AmbientLight(0.1),
# ]

# camera = Camera(vec([2, 1.7, 5]), target=vec([0, 0, 0]), vfov=17, aspect=16/9)

# # ==============================================================================
# # mesh cube

# tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
# gray = Material(vec([0.2, 0.2, 0.2]))

# # Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
# (i, p, n, t) = read_obj(open("models/cube.obj"))

# scene = Scene([
#     # Make a big sphere for the floor
#     Sphere(vec([0, -40, 0]), 39.5, gray),
# ] + [
#     Mesh(i, 0.5*p, None, None, tan),
# ])

# lights = [
#     PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#     AmbientLight(0.1),
# ]

# camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]), vfov=25, aspect=16/9)


# ==============================================================================


red = Material(vec([0.7019, 0.106, 0.106]), k_s=0.3, p=30, k_m=0.3)
white = Material(vec([-2.0, -.03, 1]), k_m=0.4)
tan = Material(vec([-2.2, 0.0, 0.0]), 0.6)

vs_list = 0.5 * read_obj_triangles(open("models/cube.obj"))
vs_list2 = 0.25 * (read_obj_triangles(open("models/cube.obj"))) + 2

# ========================================================
# tree

tree1mesh = Material(load_image("textures/flowers.jpg"))
tree2mesh = Material(load_image("textures/sunflowers.jpg"))
gray = Material(vec([0.2, 0.2, 0.2]))

i, p, n, t = read_obj(open("models/low-poly-tree-1.obj"))
tree1 = Mesh(i, 0.5*p + [[0.3, 0, 0]], n, t, tree1mesh)
tree2 = Mesh(i, 0.5*p + [[0.7, 0, 0]], n, t, tree2mesh)
tree3 = Mesh(i, 0.5*p + [[0.0, 0, 0]], n, t, tree1mesh)
tree4 = Mesh(i, 0.5*p + [[-0.5, 0, 0]], n, t, tree2mesh)
tree5 = Mesh(i, 0.5*p + [[-1.5, 0, 0]], n, t, tree1mesh)
tree6 = Mesh(i, 0.5*p + [[2.0, 0, 0]], n, t, tree2mesh)

###############
tanbunny = Material(vec([0.7, 0.7, 0.4]), 0.6)
graybunny = Material(vec([0.2, 0.2, 0.2]))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
(ib, pb, nb, tb) = read_obj(open("models/bunny.obj"))

###############

(i, p, n, t) = read_obj(open("models/buddha100k_norms.obj"))

scene = Scene([
    # Make a big sphere for the floor
    Sphere(vec([0, -60, 0]), 60., gray),
] + [Accel([tree1] + [tree2]+[tree3]+[tree4]+[tree5]+[tree6])] + [
    # Make triangle objects from the vertex coordinates
    Triangle(vs, tan) for vs in vs_list
] + [
    # Make triangle objects from the vertex coordinates
    Triangle(vs, white) for vs in vs_list2
] + [
    Accel([
        Mesh(i, p + [[-1.3, 0.2, 0]], n, None, rand_color())
    ])
] + [
    Accel([
        Mesh(ib, 0.7*pb + [[-1.4, -0.1, 0]], None, None, tanbunny),
        Mesh(ib, 0.7*pb + [[-0.7, 0.1, 0]], nb, None, tanbunny),
        Mesh(ib, 0.7*pb + [[.8, -0.1, 0]], None, None, tanbunny),
        Mesh(ib, 0.5*pb + [[.7, 0.1, 0]], nb, None, tanbunny)
    ])
])

lights = [
    PointLight(vec([12, 10, 5]), vec([500, 500, 500])),
    AmbientLight(0.1),
]

camera = Camera(vec([3, 1, 5]), target=vec([0, 1, 0]), vfov=30, aspect=16/9)

render(camera, scene, lights)

# ==============================================================================


# scene = Scene([
#     Sphere(vec([-0.7, 0, 0]), 0.5, red),
#     Sphere(vec([0.8, -20, 0]), 0.5, white),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
#     Triangle(vs_list[0], white)
# ])

# scene = Scene([
#     # Make a big sphere for the floor
#     Sphere(vec([-0.7, 0, 0]), 0.5, red),
#     Sphere(vec([0.8, -20, 0]), 0.5, white),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
#     Triangle(vs_list[0], white)
# ] + [
#     # Make triangle objects from the vertex coordinates
#     Triangle(vs, tan) for vs in vs_list
# ])

# scene = Scene([
#     # Make a big sphere for the floor
#     Sphere(vec([-0.7, 0, 0]), 0.5, red),
#     Sphere(vec([0.8, -20, 0]), 0.5, white),
#     Sphere(vec([0, -40, 0]), 39.5, gray),
#     Triangle(vs_list[0], white),
# ] + [
#     # Make triangle objects from the vertex coordinates
#     Triangle(vs, tan) for vs in vs_list
# ] + [
#     # Make triangle objects from the vertex coordinates
#     Triangle(vs, tan) for vs in vs_list2
# ] + [
#     # Make a big sphere for the floor
#     Sphere(vec([0, -60, 0]), 60., gray),
# ] + [
#     tree1
# ]
# )


# lights = [
#     PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
#     AmbientLight(0.1)
# ]

# camera = Camera(vec([3, 4, 5]), target=vec(
#     [0, -0.4, 0]), vfov=24, aspect=16/9)

# render(camera, scene, lights)
