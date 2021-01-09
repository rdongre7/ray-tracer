import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : float, (3,) or (h,w,3) -- the diffuse coefficient
          k_s : float, (3,) or (h,w,3) -- the specular coefficient
          p : float or (h,w) -- the specular exponent
          k_m : float, (3,) or (h,w,3) -- the mirror reflection coefficient
          k_a : float, (3,) or (h,w,3) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d

    def lookup(self, param, uv):
        if isinstance(param, float) or len(param.shape) == 1:
            return param
        else:
            u = int(uv[1] * (param.shape[0] - 1))
            v = int(uv[0] * (param.shape[1] - 1))
            return param[u][v]


class Hit:

    def __init__(self, t, point=None, normal=None, uv=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          uv : (2,) -- the texture coordinates at the intersection point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.uv = uv
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(ray.direction, ray.origin - self.center)
        c = np.dot(ray.origin - self.center,
                   ray.origin - self.center) - self.radius ** 2.0
        d = b**2.0 - 4.0*a*c
        if d >= 0:
            t_values = [t for t in [(b + np.sqrt(d)) / (-2.0 * a),
                                    (b - np.sqrt(d)) / (-2.0 * a)
                                    ] if t >= ray.start and t <= ray.end]
            if len(t_values) > 0:
                t = min(t_values)
                pt = ray.origin + t * ray.direction
                normal = normalize(pt - self.center)
                normal_uv = normalize(
                    self.center - vec([pt[0], pt[1], -1 * pt[2]]))
                u = 0.5 + \
                    (np.arctan2(-1 * normal_uv[0], normal_uv[2]) / (2 * np.pi))
                v = 0.5 - (np.arcsin(normal_uv[1]) / np.pi)
                uv = vec([u, v])
                return Hit(t, pt, normal, uv, self.material)
        return no_hit


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        def check_sign(a, b, c, d):
            return np.sign(np.dot(np.cross(b-a, c-a), d-a))

        start_pt = ray.origin
        end_pt = ray.origin + ray.direction

        if check_sign(start_pt, end_pt, self.vs[0], self.vs[1]) == check_sign(
                start_pt, end_pt, self.vs[1], self.vs[2]) and check_sign(
                start_pt, end_pt, self.vs[1], self.vs[2]) == check_sign(
                start_pt, end_pt, self.vs[2], self.vs[0]):
            normal = np.cross(self.vs[1] - self.vs[0], self.vs[2] - self.vs[0])
            t = np.dot(self.vs[0] - ray.origin, normal) / \
                np.dot(ray.direction, normal)
            if t >= ray.start and t <= ray.end:
                pt = ray.origin + t * ray.direction
                d = np.linalg.det(vec([self.vs[1] - self.vs[0], self.vs[1] -
                                       self.vs[2], ray.direction]))
                if d != 0:
                    pt1 = vec([self.vs[1] - pt, self.vs[1] -
                               self.vs[2], ray.direction])
                    pt2 = vec([self.vs[1] - self.vs[0],
                               self.vs[1] - pt, ray.direction])

                    beta = np.linalg.det(pt1)/d
                    gamma = np.linalg.det(pt2)/d
                    if beta >= 0 and gamma >= 0:
                        uv = vec([beta, gamma])
                        return Hit(t, pt, normalize(normal), uv, self.material)
        return no_hit


class Mesh:

    def __init__(self, inds, posns, normals, uvs, material):
        self.inds = np.array(inds, np.int32)
        self.posns = np.array(posns, np.float32)
        self.normals = np.array(
            normals, np.float32) if normals is not None else None
        self.uvs = np.array(uvs, np.float32) if uvs is not None else None
        self.material = material

    def extract_triangle(self, val):
        return vec([self.posns[val[0]],
                    self.posns[val[1]], self.posns[val[2]]])

    def intersect(self, ray, triangle=None, t=None, beta=None, gamma=None, i=None):
        """Computes the intersection between a ray and this mesh, if it exists.
           Use the batch_intersect function in the utils package

        Parameters:
          ray : Ray -- the ray to intersect with the mesh
        Return:
          Hit -- the hit data
        """
        if triangle is None:
            triangles = np.zeros((len(self.inds), 3, 3), np.float32)
            for j, val in enumerate(self.inds):
                triangles[j] = self.extract_triangle(val)
            t, beta, gamma, i = batch_intersect(triangles, ray)
            triangle = triangles[i]
        if i != -1 and t >= ray.start and t <= ray.end:
            pt = ray.origin + t * ray.direction
            normal = np.cross(triangle[1] - triangle[0],
                              triangle[2] - triangle[0])
            uv = vec([beta, gamma])
            if self.normals is None and self.uvs is None:
                return Hit(t, pt, normalize(normal), uv, self.material)
            d = np.linalg.det(vec([triangle[1] - triangle[0], triangle[1] -
                                   triangle[2], ray.direction]))
            if d != 0:
                pt1 = vec([triangle[1] - pt, triangle[1] -
                           triangle[2], ray.direction])
                pt2 = vec([triangle[1] - triangle[0],
                           triangle[1] - pt, ray.direction])
                b = np.linalg.det(pt1)/d
                g = np.linalg.det(pt2)/d
                a = 1 - b - g
                if b >= 0 and g >= 0:
                    if self.normals is not None:
                        normal = b * self.normals[self.inds[i][0]] + a * \
                            self.normals[self.inds[i][1]] + \
                            g * self.normals[self.inds[i][2]]
                    if self.uvs is not None:
                        uv = b * self.uvs[self.inds[i][0]] + a * \
                            self.uvs[self.inds[i][1]] + \
                            g * self.uvs[self.inds[i][2]]
                    return Hit(t, pt, normalize(normal), uv, self.material)
        return no_hit


class Node:
    def __init__(self, min, max, left=None, right=None, start=None, end=None):
        self.min = min
        self.max = max
        self.left = left
        self.right = right
        self.start = start
        self.end = end

    def print_tree(self):
        if self.left is None:
            print(self.start)
            print(self.end)
        else:
            self.left.print_tree()
            self.right.print_tree()


class Accel:
    def __init__(self, meshes, n_triangles=100):
        self.triangles = []
        self.info = []
        self.n_triangles = n_triangles
        for mesh in meshes:
            for i, val in enumerate(mesh.inds):
                self.info.append([mesh, i])
                self.triangles.append(mesh.extract_triangle(val))
        self.root = self.build_tree(0, len(self.triangles))

    def build_tree(self, start, end):
        sliced_triangles = self.triangles[start:end]
        bb = self.bounding_box(sliced_triangles)
        if end - start <= self.n_triangles:
            return Node(bb[0], bb[1], start=start, end=end)
        else:
            sliced_info = self.info[start:end]
            axis = -1
            diff = -1
            for ax in range(3):
                temp = bb[1][ax] - bb[0][ax]
                if temp > diff:
                    axis = ax
                    diff = temp
            sorted_triangles = sorted(zip(sliced_triangles, sliced_info),
                                      key=lambda pair: self.bounding_box_triangle(pair[0])[0][axis])
            for i in range(len(sorted_triangles)):
                self.triangles[i + start] = sorted_triangles[i][0]
                self.info[i + start] = sorted_triangles[i][1]
            med = int((start + end) / 2)
            return Node(bb[0], bb[1], left=self.build_tree(start, med),
                        right=self.build_tree(med, end))

    def bounding_box_triangle(self, triangle):
        min_val = [None] * 3
        max_val = [None] * 3
        for i in range(3):
            for point in triangle:
                if min_val[i] is None or min_val[i] > point[i]:
                    min_val[i] = point[i]
                if max_val[i] is None or max_val[i] < point[i]:
                    max_val[i] = point[i]
        return [min_val, max_val]

    def bounding_box(self, triangles):
        min_val = [None] * 3
        max_val = [None] * 3
        for triangle in triangles:
            for i in range(3):
                for point in triangle:
                    if min_val[i] is None or min_val[i] > point[i]:
                        min_val[i] = point[i]
                    if max_val[i] is None or max_val[i] < point[i]:
                        max_val[i] = point[i]
        for i in range(3):
            min_val[i] -= 1e-6
            max_val[i] += 1e-6
        return [min_val, max_val]

    def miss_bb(self, ray, min_val, max_val):
        t1 = (min_val[0] - ray.origin[0]) / \
            ray.direction[0] if ray.direction[0] != 0 else 0
        t2 = (max_val[0] - ray.origin[0]) / \
            ray.direction[0] if ray.direction[0] != 0 else 0
        t3 = (min_val[1] - ray.origin[1]) / \
            ray.direction[1] if ray.direction[1] != 0 else 0
        t4 = (max_val[1] - ray.origin[1]) / \
            ray.direction[1] if ray.direction[1] != 0 else 0
        t5 = (min_val[2] - ray.origin[2]) / \
            ray.direction[2] if ray.direction[2] != 0 else 0
        t6 = (max_val[2] - ray.origin[2]) / \
            ray.direction[2] if ray.direction[2] != 0 else 0
        tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
        tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))
        return tmin > tmax or tmin < ray.start or tmin > ray.end

    def intersect_helper(self, ray, root):
        if self.miss_bb(ray, root.min, root.max):
            return no_hit
        else:
            if root.start is not None:
                triangles = vec(self.triangles[root.start:root.end])
                t, beta, gamma, i = batch_intersect(
                    triangles, ray)
                if i == -1:
                    return no_hit
                mesh = self.info[root.start + i][0]
                new_i = self.info[root.start + i][1]
                hit = mesh.intersect(
                    ray, triangles[i], t, beta, gamma, new_i)
                return hit
            else:
                left = self.intersect_helper(ray, root.left)
                right = self.intersect_helper(ray, root.right)
                if left.t < right.t:
                    return left
                return right

    def intersect(self, ray):
        return self.intersect_helper(ray, self.root)


class Camera:

    def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        d = target - eye
        self.w = -normalize(d)
        self.u = normalize(np.cross(up, self.w))
        self.v = normalize(np.cross(self.w, self.u))
        self.d = np.linalg.norm(d)
        self.height = 2*self.d*np.tan(vfov/2 * np.pi/180)
        self.width = aspect * self.height

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                      corner of the image and (1,1) is the upper right
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        img_point -= 0.5
        u = self.width * img_point[0]
        v = self.height * img_point[1]
        direction = -self.d * self.w + u * self.u + v * self.v
        return Ray(self.eye, direction)


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        l = normalize(self.position - hit.point)
        v = normalize(ray.origin - hit.point)
        h = normalize(v + l)
        r = np.linalg.norm(self.position - hit.point)
        d = max(0, np.dot(hit.normal, l)) / r ** 2.0
        k_d = hit.material.lookup(hit.material.k_d, hit.uv)
        k_s = hit.material.lookup(hit.material.k_s, hit.uv)
        p = hit.material.lookup(hit.material.p, hit.uv)
        s = k_d + k_s * (max(0, np.dot(hit.normal, h)))**p
        return d * s * self.intensity


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        k_a = hit.material.lookup(hit.material.k_a, hit.uv)
        return self.intensity * k_a


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle,  Mesh] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        hits = [surf.intersect(ray) for surf in self.surfs]
        return min(hits, key=lambda hit: hit.t)


MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    color = scene.bg_color
    if hit.point is not None:
        color = np.zeros(3)
        for light in lights:
            if isinstance(light, AmbientLight):
                color += light.illuminate(ray, hit, scene)
            else:
                ray_2 = Ray(hit.point, light.position - hit.point, 0.000001)
                is_not_shadowed = scene.intersect(ray_2).point is None
                if is_not_shadowed:
                    color += light.illuminate(ray, hit, scene)
        if depth < MAX_DEPTH - 1:
            v = normalize(ray.origin - hit.point)
            r = normalize(2.0 * np.dot(hit.normal, v) * hit.normal - v)
            ray = Ray(hit.point, r, 0.000001)
            k_m = hit.material.lookup(hit.material.k_m, hit.uv)
            return color + k_m * shade(ray, scene.intersect(ray),
                                       scene, lights, depth + 1)
    return color


def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    img = np.zeros((ny, nx, 3), np.float32)
    for i in range(ny - 1):
        for j in range(nx - 1):
            x_pos = j / nx
            y_pos = i / ny
            ray = camera.generate_ray(vec([x_pos, y_pos]))
            hit = scene.intersect(ray)
            img[i][j] = shade(ray, hit, scene, lights)
    return img
