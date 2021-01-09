from utils import *
from ray import *
from cli import render
import time

rng = np.random.default_rng(0)

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]))


def rand_color():
    return Material(rng.random(3), 0.6)


print("loading mesh...")
# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
(i, p, n, t) = read_obj(open("models/buddha100k_norms.obj"))


def rot_y(p, theta):
    return p @ [[np.cos(theta)], 0, -np.sin(theta), [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]]


def rand_rot(p):
    return rot_y(p, rng.random() - 0.5)


print("building scene...")
start = time.time()
scene = Scene([
    # Make a big sphere for the floor
    Sphere(vec([0, -100, 0]), 99.5, gray),
] + [
    Accel([
        Mesh(i, p + [[-1.3, 0.2, 0]], n, None, rand_color()),
        Mesh(i, p + [[0.0, 0.5, 0]], n, None, rand_color()),
        Mesh(i, p + [[1.3, 0.2, 0]], n, None, rand_color()),
        Mesh(i, p + [[0.65, 0.2, 0.866*1.3]], n, None, rand_color()),
        Mesh(i, p + [[-0.65, 0.2, 0.866*1.3]], n, None, rand_color()),
        Mesh(i, p + [[0.65, 0.2, -0.866*1.3]], n, None, rand_color()),
        Mesh(i, p + [[-0.65, 0.2, -0.866*1.3]], n, None, rand_color())
    ])
])

lights = [
    PointLight(vec([10, 12, 7]), vec([300, 300, 300])),
    AmbientLight(0.1),
]

camera = Camera(vec([0.8, 1.8, 5]), target=vec(
    [0, 0.5, 0]), vfov=15, aspect=16/9)

print("rendering...")
render(camera, scene, lights)

print("Time taken: " + str(time.time() - start))
