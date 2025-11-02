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
        self.prisms = []          # Holds RectangularPrism objects
        self.collision_markers = []  # Optional spheres for visualization
    # keep initialization minimal; collision uses center-line segments
    # between successive joint transforms (legacy behavior).

    def add_scene_prisms(self, obj):
        """
        Convert a Cuboid from Swift into a RectangularPrism for collision detection
        """
        if not isinstance(obj, Cuboid):
            return

        lwh = obj.scale
        transform = obj.T         # 4x4 numpy array
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
        """
        Check collision for a robot at one or multiple joint states
        q_matrix: list or np.ndarray of joint configurations
        """
        q_matrix = [q_matrix]

        # Clear previous visual markers when visualising
        if self.visualise:
            self._clear_collision_markers()

        intersections = []

        for q in q_matrix:
            tr_all = robot.fkine_all(q)
            tr_all_arr = None

            A = np.asarray(tr_all.A)

            # If A has shape (4,4,n) transpose to (n,4,4)
            if A.ndim == 3 and A.shape[0] == 4 and A.shape[1] == 4:
                tr_all_arr = np.transpose(A, (2, 0, 1))
            elif A.ndim == 2 and A.shape == (4, 4):
                tr_all_arr = A.reshape((1, 4, 4))
            else:
                tr_all_arr = np.asarray(A)

            # Go link by link (legacy behaviour: single segment between joint transforms)
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
        """
        Check if a line segment (link) intersects a rectangular prism
        """
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

    def _sample_link_segments(self, start, end, n_along=3, n_around=8, radius=0.05):
        """Approximate a cylindrical/offset link by sampling multiple short
        line segments around the center-line.

        Returns a list of (start,end) pairs (each a 3-vector).
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        vec = end - start
        length = np.linalg.norm(vec)
        if length == 0:
            return []
        dir_unit = vec / length

        # find two orthonormal vectors perpendicular to dir_unit
        # pick arbitrary vector not parallel to dir_unit
        arb = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arb, dir_unit)) > 0.9:
            arb = np.array([0.0, 1.0, 0.0])
        u = np.cross(dir_unit, arb)
        u = u / np.linalg.norm(u)
        v = np.cross(dir_unit, u)
        v = v / np.linalg.norm(v)

        segs = []
        # include center-line sampling as well
        t_vals = np.linspace(0.0, 1.0, n_along)
        for a in range(n_around):
            theta = 2.0 * np.pi * (a / float(n_around))
            offset = radius * (np.cos(theta) * u + np.sin(theta) * v)
            pts = [start + dir_unit * (t * length) + offset for t in t_vals]
            # build short segments between successive points
            for k in range(len(pts) - 1):
                segs.append((pts[k], pts[k + 1]))
        # also add the pure centerline segments (no offset)
        center_pts = [start + dir_unit * (t * length) for t in t_vals]
        for k in range(len(center_pts) - 1):
            segs.append((center_pts[k], center_pts[k + 1]))

        return segs

    def _add_collision_marker(self, point, size=0.02, color=[1, 0, 0, 1]):

        # Adds a 3d marker to a collision point

        if self.env is None:
            return

        sx = sy = sz = float(size)
        # create cuboid centered at point
        cub = Cuboid(scale=[sx, sy, sz], color=[float(c) for c in color], pose=SE3(float(point[0]), float(point[1]), float(point[2])))

        # Use the socket send path (blocking) to register the shape with Swift.
        # This mirrors the earlier behaviour where the call could block the UI
        # but reliably caused the visible marker to appear in Swift.
        try:
            shape_dict = cub.to_dict()

            # blocking send to Swift; older Swift envs expect this
            self.env._send_socket("shape", [shape_dict])

            # record locally so we can remove later
            self.collision_markers.append(cub)
            try:
                x, y, z = float(point[0]), float(point[1]), float(point[2])
                print(f"Added collision marker at x={x:.3f}, y={y:.3f}, z={z:.3f}")
            except Exception:
                print("Added collision marker at (unknown coords)")
        except Exception:
            # If the blocking send fails, report but allow the exception to
            # propagate so any Swift-side issues are visible to the caller.
            try:
                x, y, z = float(point[0]), float(point[1]), float(point[2])
                print(f"(socket send failed) collision marker intended at x={x:.3f}, y={y:.3f}, z={z:.3f}")
            except Exception:
                print("(socket send failed) collision marker intended at (unknown coords)")
            raise

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
    """
    Check if a 3D point is inside a triangle using barycentric coordinates
    """
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]
    w = intersect_p - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv
    if abs(D) < 1e-12:
        return False

    s = (uv * wv - vv * wu) / D
    t = (uv * wu - uu * wv) / D
    eps = 1e-9
    if s < -eps or s > 1.0 + eps:
        return False
    if t < -eps or (s + t) > 1.0 + eps:
        return False

    return True
