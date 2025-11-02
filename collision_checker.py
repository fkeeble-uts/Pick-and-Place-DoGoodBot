# collision_checker.py
import numpy as np
from itertools import combinations
from spatialgeometry import Cuboid
from spatialmath import SE3
from ir_support import RectangularPrism, line_plane_intersection

class CollisionChecker:
    def __init__(self, env=None, visualise=False):
        self.env = env
        self.visualise = visualise
        self.prisms = []
        self.collision_markers = []

    def add_scene_prisms(self, obj):
        """
        Convert a Cuboid from Swift into a RectangularPrism for collision detection
        """

        lwh = obj.scale
        transform = obj.T
        center = transform[:3, 3] # x, y, z position
        prism = RectangularPrism(lwh[0], lwh[1], lwh[2], center=center)
        vertices, faces, face_normals = prism.get_data()
        self.prisms.append({
            'vertices': vertices,
            'faces': faces,
            'face_normals': face_normals,
            'original_obj': obj
        })

    def check_collision_for_q(self, robot, q_matrix, return_all=False):
        # Checks the robot at a joint configuration for collisions

        q_matrix = [q_matrix]

        # Clear previous visual markers when visualising
        if self.visualise:
            self._clear_collision_markers()

        intersections = []

        for q in q_matrix:
            tr_all = robot.fkine_all(q)

            A = np.asarray(tr_all.A)

            tr_all_arr = np.asarray(A)

            # Go link by link
            n_transforms = tr_all_arr.shape[0]
            for i in range(n_transforms - 1):
                p_start = tr_all_arr[i][:3, 3]
                p_end = tr_all_arr[i + 1][:3, 3]
                for prism in self.prisms:
                    intersect_p = self._link_intersects_prism(p_start, p_end, prism)
                    if intersect_p is not None:
                        # record intersection (dedupe nearby points)
                        ip = np.asarray(intersect_p, dtype=float)
                        keep = True
                        for ex in intersections:
                            if np.linalg.norm(ex - ip) < 1e-6:
                                keep = False
                                break
                        if keep:
                            intersections.append(ip)

                        # Add a visual marker if requested and env is available
                        if self.visualise and self.env is not None:
                            try:
                                self._add_collision_marker(ip)
                            except Exception:
                                # Don't let visualization errors stop collision detection
                                pass
                        # If not returning all, stop at first intersection
                        if not return_all:
                            return True
        if return_all:
            return intersections
        return False

    def _link_intersects_prism(self, start, end, prism):
        # Checks if a line segment intersects a rectangular prism

        vertices = prism['vertices']
        faces = prism['faces']
        normals = prism['face_normals']

        for j, face in enumerate(faces):
            vert_on_plane = vertices[face][0]
            intersect_p, check = line_plane_intersection(normals[j], vert_on_plane, start, end)
            if check != 1:
                continue
            triangle_list = np.array(list(combinations(face, 3)), dtype=int)
            for tri in triangle_list:
                inside = is_intersection_point_inside_triangle(intersect_p, vertices[tri])
                if inside:
                    return np.asarray(intersect_p)
        return None

    def _add_collision_marker(self, point, size=0.02, color=[1, 0, 0, 1]):
        # Adds a 3d marker to a collision point

        sx = sy = sz = float(size)
        # create cuboid centered at point
        cub = Cuboid(scale=[sx, sy, sz], color=[float(c) for c in color], pose=SE3(float(point[0]), float(point[1]), float(point[2])))

        # Use the socket send path (blocking) to register the shape with Swift.
        # This mirrors the earlier behaviour where the call could block the UI
        # but reliably caused the visible marker to appear in Swift.
        shape_dict = cub.to_dict()

        # blocking send to Swift
        self.env._send_socket("shape", [shape_dict])

        # record locally so we can remove later
        self.collision_markers.append(cub)

        x, y, z = float(point[0]), float(point[1]), float(point[2])
        print(f"Added collision marker at x={x:.3f}, y={y:.3f}, z={z:.3f}")

    def _clear_collision_markers(self):
        """Remove previously added collision markers from the env (if any)."""
        if self.env is None:
            self.collision_markers = []
            return
        for m in list(self.collision_markers):
            try:
                self.env.remove(m)
            except Exception:
                pass
        self.collision_markers = []


def is_intersection_point_inside_triangle(intersect_p, triangle_verts):
    # Check if a point lies within a triangle (from lab 5)
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv

    # Get and test parametric coords (s and t)
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0:  # intersect_p is outside Triangle
        return 0

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0:  # intersect_p is outside Triangle
        return False

    return True  # intersect_p is in Triangle
