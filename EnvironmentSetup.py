import swift
from spatialmath import SE3
from spatialgeometry import Cylinder, Cuboid
from math import pi

class Scene:
    """
    A class to hold all configuration variables for the simulation environment.
    """
    def __init__(self, env):
        self.env = env

        # ----------------------------------------------------
        # CONSTANTS & CONFIGURATION
        # ----------------------------------------------------

        # --- Environment & Timing ---
        self.SIM_STEP_TIME = 0.02
        self.TRAJ_STEPS = 60

        # --- Shared Object Dimensions ---
        self.BUTTON_RADIUS = 0.05
        self.BUTTON_HEIGHT = 0.03
        self.BUTTON_BASE_LENGTH = 0.12
        self.BUTTON_BASE_WIDTH = 0.12
        self.BUTTON_BASE_HEIGHT = 0.02

        # --- Gripper Visual Dimensions ---
        self.FINGER_LENGTH = 0.08
        self.FINGER_THICKNESS = 0.01
        self.FINGER_HEIGHT = 0.08
        self.FINGER_GAP_OPEN = 0.06
        self.FINGER_GAP_CLOSED = 0.055
        self.FINGER_BACK_OFFSET = 0.02
        self.FINGER_Z_OFFSET = -0.03
        
        # --- Global State Variable (used by R1's gripper) ---
        self._finger_gap_r1 = self.FINGER_GAP_OPEN

        # ----------------------------------------------------
        # SWIFT OBJECT PARAMETERS
        # ----------------------------------------------------

        self.wall_height = 2.5
        self.wall_thickness = 0.05
        self.floor_height = 0.01

        # Table 1 (large workstation table)
        self.table1_length = 4.0
        self.table1_width  = 0.75
        self.table1_height = 1.0
        table1_offset_from_wall = 0.5
        self.table1_center_y = -1.5 + self.wall_thickness + table1_offset_from_wall + self.table1_width / 2

        # Table 2 (smaller / front table)
        self.table2_length = 0.3
        self.table2_width  = 0.8
        self.table2_height = 1.0
        table2_spacing = 0.6
        self.table2_center_y = self.table1_center_y + (self.table1_width / 2) + table2_spacing + (self.table2_width / 2)

        # Table 3 (drinks shelf)
        self.table3_length = 4.0
        self.table3_width  = 0.325
        self.table3_height = 1.2
        self.table3_center_y = -1.75 + self.wall_thickness + self.table3_width / 2

        # Glass table (for cups)
        self.glass_table_length = 0.4
        self.glass_table_width  = 0.7
        self.glass_table_height = 1.05
        self.glass_table_center_x = self.table1_length/2 + self.glass_table_length/2 + 0.1
        self.glass_table_center_y = self.table1_center_y

        # Ingredients table
        self.ingredients_table_length = 0.4
        self.ingredients_table_width  = 0.7
        self.ingredients_table_height = 1.05
        self.ingredients_table_center_x = -(self.table1_length/2 + self.ingredients_table_length/2 + 0.1)
        self.ingredients_table_center_y = self.table1_center_y

        # Serving Pedestals 
        self.pedestal_length = 0.2
        self.pedestal_width = 0.2
        self.pedestal_height = 1.0
        self.pedestal_x_offset = 0.5
        self.pedestal_y_offset = 0.3

        # Control Desk
        self.control_desk_length = 0.6
        self.control_desk_width  = 0.8
        self.control_desk_height = 1.0
        self.control_desk_center_x = 2.7
        self.control_desk_center_y = self.table2_center_y + 1.5

        # Emergency stop button
        self.button_center_x = self.control_desk_center_x - 0.2
        self.button_center_y = (self.table2_center_y + self.control_desk_width / 2 - 0.1) + 1.5
        self.button_center_z = self.table2_height + self.BUTTON_BASE_HEIGHT / 2 -0.01

        # LED / colour parameters
        self.base_color      = [0.1, 0.1, 0.15, 1]
        self.top_color       = [0.0, 0.6, 0.8, 1]
        self.button_base_color = [0.2, 0.2, 0.2, 1]
        self.button_color = [0.8, 0, 0, 1]
        self.top_glow_color  = [0.0, 0.8, 0.8, 0.3]
        self.led_color       = [0.0, 0.8, 0.8, 0.6]
        self.led_height      = 0.05
        self.led_offset      = 0.01
        self.led_margin      = 0.02
        self.num_wraps       = 3
        self.wrap_spacing_factor = 0.25

        # --- Bar Mat Configuration  ---
        self.BAR_MAT_THICKNESS = 0.02 
        self.BAR_MAT_LENGTH_X = 0.2                
        self.BAR_MAT_WIDTH_Y = 0.2                   
        self.BAR_MAT_COLOR = [0.1, 0.1, 0.1, 0.8]  
        self.BAR_MAT_Z_POS = self.table1_height + self.floor_height + self.BAR_MAT_THICKNESS / 2

        # Mat Positions on Table 1 
        self.BAR_MAT_POSITIONS = [
            # Mat A: Between R1 and R2 
            {"name": "Mat_A_R1_R2", "x": 1.1, "y": self.table1_center_y}, 
            # Mat B: Between R2  and R3 (Drinkbot) 
            {"name": "Mat_B_R2_R3", "x": -1.0, "y": self.table1_center_y},
            # Mat C: On UR3 Desk 
            {"name": "Mat_UR3", "x": 0.0, "y": self.table2_center_y-0.25},

        ]

        # --- Robot base poses ---
        self.ROBOT_BASE_POSES = {
            "R1_ICE_GLASS": SE3(1.6, float(self.table1_center_y), float(self.table1_height + self.floor_height)),
            "R2_ALCOHOL":   SE3(0.0, float(self.table1_center_y), float(self.table1_height + self.floor_height)),
            "R3_MIXERS":    SE3(-1.6, float(self.table1_center_y), float(self.table1_height + self.floor_height )),
            "R4_SERVER":    SE3(0, float(self.table2_center_y+ 0.2), float(self.table2_height + self.floor_height - 0.009)),
        }

        # ----------------------------------------------------
        # INITIAL ENVIRONMENT OBJECTS
        # ----------------------------------------------------
        floor = Cuboid(scale=[6, 5, 0.02],
                       color=[0.25, 0.3, 0.35, 1],
                       pose=SE3(0, 00.75, self.floor_height))
        self.env.add(floor)

        # --- Walls ---
        back_wall = Cuboid(scale=[6, self.wall_thickness, self.wall_height],
                           color=[0.85, 0.85, 0.9, 1],
                           pose=SE3(0, -1.75, self.wall_height/2))
        self.env.add(back_wall)

        left_wall = Cuboid(scale=[self.wall_thickness, 5, self.wall_height],
                           color=[0.85, 0.85, 0.9, 1],
                           pose=SE3(-3, 0.75, self.wall_height/2))
        self.env.add(left_wall)

        right_wall = Cuboid(scale=[self.wall_thickness, 5, self.wall_height],
                            color=[0.85, 0.85, 0.9, 1],
                            pose=SE3(3, 0.75, self.wall_height/2))
        self.env.add(right_wall)

        front_wall = Cuboid (scale=[6, self.wall_thickness, self.wall_height],
                            color=[0.85, 0.85, 0.9, 0.2],
                            pose=SE3(0, 1.5, self.wall_height/2))
        self.env.add(front_wall)


        # --- Tables ---
        tables = [
            {"name": "Workstation", "length": self.table1_length, "width": self.table1_width, "height": self.table1_height, "center": SE3(0, self.table1_center_y, 0), "leds": False},
            {"name": "UR3e Table", "length": self.table2_length, "width": self.table2_width, "height": self.table2_height, "center": SE3(0, self.table2_center_y, 0), "leds": True},
            {"name": "Glass Table", "length": self.glass_table_length, "width": self.glass_table_width, "height": self.glass_table_height, "center": SE3(self.glass_table_center_x, self.glass_table_center_y, 0), "leds": True},
            {"name": "Drinks Shelf", "length": self.table3_length, "width": self.table3_width, "height": self.table3_height, "center": SE3(0, self.table3_center_y, 0), "leds": False},
            {"name": "Ingredients Table", "length": self.ingredients_table_length, "width": self.ingredients_table_width, "height": self.ingredients_table_height, "center": SE3(self.ingredients_table_center_x, self.ingredients_table_center_y, 0), "leds": True},
            {"name": "Serving Pedestal Left", "length": self.pedestal_length, "width": self.pedestal_width, "height": self.pedestal_height,"center": SE3(-self.pedestal_x_offset, self.table2_center_y + self.pedestal_y_offset, 0), "leds": True},
            {"name": "Serving Pedestal Right", "length": self.pedestal_length, "width": self.pedestal_width, "height": self.pedestal_height,"center": SE3(self.pedestal_x_offset, self.table2_center_y + self.pedestal_y_offset, 0), "leds": True},
            {"name": "Control Desk", "length": self.control_desk_length, "width": self.control_desk_width, "height": self.control_desk_height, "center": SE3(self.control_desk_center_x, self.control_desk_center_y, 0), "leds": False},    
        ]

        for i, t in enumerate(tables):
            cx, cy, cz = t["center"].t
            h, l, w = t["height"], t["length"], t["width"]
            base = Cuboid(scale=[l, w, h-0.05], color=self.base_color, pose=SE3(cx, cy, (h-0.05)/2))
            self.env.add(base)
            top = Cuboid(scale=[l, w, 0.05], color=self.top_color, pose=SE3(cx, cy, h - 0.025))
            self.env.add(top)
            glow_scale_x = l*1.05 if l <= self.table2_length else l
            glow_scale_y = w*1.05 if l <= self.table2_length else w
            glow = Cuboid(scale=[glow_scale_x, glow_scale_y, 0.02], color=self.top_glow_color, pose=SE3(cx, cy, h - 0.015))
            self.env.add(glow)
            if t["leds"]:
                led = Cuboid(scale=[l + self.led_margin*2, w + self.led_margin*2, self.led_height], color=self.led_color, pose=SE3(cx, cy, (self.led_height/2)+self.led_offset))
                self.env.add(led)
                for j in range(1, self.num_wraps+1):
                    wrap_z = self.led_height + (h - 0.05 - self.led_height) * self.wrap_spacing_factor * j
                    wrap_ring = Cuboid(scale=[l + self.led_margin*2, w + self.led_margin*2, self.led_height], color=self.led_color, pose=SE3(cx, cy, wrap_z))
                    self.env.add(wrap_ring)

        # --- Bar Mats ---
        for mat_config in self.BAR_MAT_POSITIONS:
            mat = Cuboid(
                scale=[self.BAR_MAT_LENGTH_X, self.BAR_MAT_WIDTH_Y, self.BAR_MAT_THICKNESS],
                color=self.BAR_MAT_COLOR,
                pose=SE3(mat_config["x"], mat_config["y"], self.BAR_MAT_Z_POS)
            )
            self.env.add(mat)

        # --- Emergency stop button ---
        stop_base = Cuboid(scale=[self.BUTTON_BASE_LENGTH, 
                                  self.BUTTON_BASE_WIDTH, self.BUTTON_BASE_HEIGHT], 
                                  color=self.button_base_color, 
                                  pose=SE3(self.button_center_x, self.button_center_y, 
                                           self.button_center_z + self.BUTTON_BASE_HEIGHT/2))
        env.add(stop_base)

        red_button = Cylinder(radius=self.BUTTON_RADIUS, 
                              length=self.BUTTON_HEIGHT, 
                              color=self.button_color, 
                              pose=SE3(self.button_center_x, self.button_center_y, 
                                       self.button_center_z + self.BUTTON_BASE_HEIGHT/2 + self.BUTTON_HEIGHT/2))
        env.add(red_button)

        # --- Glasses ---
        self.glass_objects = []
        self.glass_poses = []
        glass_radius = 0.05
        self.glass_height = 0.2
        glass_color = [0.8, 0.8, 0.8, 0.6]

        width_fractions = [0.1, 0.5, 0.9]
        length_fractions = [0.1, 0.5, 0.9]

        for yf in width_fractions:
            for xf in length_fractions:
                x_pos = self.glass_table_center_x - self.glass_table_length/2 + xf * self.glass_table_length
                y_pos = self.glass_table_center_y - self.glass_table_width/2 + yf * self.glass_table_width
                z_pos = self.glass_table_height + self.glass_height / 2

                glass = Cylinder(radius=glass_radius,
                                 length=self.glass_height,
                                 color=glass_color,
                                 pose=SE3(x_pos, y_pos, z_pos))
                self.env.add(glass)
                self.glass_objects.append(glass)
                self.glass_poses.append(SE3(x_pos, y_pos, z_pos))

        # --- Drinks on drink shelf ---
        self.drink_objects = []
        self.drink_poses = []
        self.drink_radius = 0.05
        self.drink_height = 0.2
        drink_color = [0, 0, 0.4, 0.7]
        drink_count = 9
        drink_gaps = (2 - self.drink_radius) / drink_count + 0.03

        for i in range(drink_count):
            pose = SE3(1 - drink_gaps * i, self.table3_center_y, 
                       self.table3_height + self.drink_height/2)
            drink = Cylinder(radius=self.drink_radius,
                             length=self.drink_height,
                             color=drink_color,
                             pose=pose)
            self.env.add(drink)
            self.drink_objects.append(drink)
            self.drink_poses.append(pose)

        # --- Ingredients Table Objects ---
        self.ingredient_objects = []
        self.cube_objects = []
        self.cube_poses = []
        board_length, board_width, board_height = 0.3, 0.185, 0.02
        cube_size = 0.025
        cube_spacing_x = 0.07
        cube_spacing_y = 0.035
        board_color = [1.0, 1.0, 1.0, 1.0]
        cube_colors = {
            "yellow": [1.0, 1.0, 0.0, 1.0],
            "green":  [0.0, 1.0, 0.0, 1.0],
            "blue":   [0.0, 0.0, 1.0, 1.0],
        }
        
        for i, yf in enumerate([0.2, 0.5, 0.8]):
            x_pos = (self.ingredients_table_center_x - self.ingredients_table_length/2 + 0.5 * self.ingredients_table_length)
            y_pos = (self.ingredients_table_center_y - self.ingredients_table_width/2 + yf * self.ingredients_table_width)
            z_pos = self.ingredients_table_height + board_height / 2
            board = Cuboid(scale=[board_length, board_width, board_height], color=board_color, pose=SE3(x_pos, y_pos, z_pos))
            self.env.add(board)
            self.ingredient_objects.append(board)

            color_key = list(cube_colors.keys())[i]
            x_start = x_pos - (3 * cube_size + 2 * cube_spacing_x) / 2 + cube_size / 2
            y_start = y_pos - (3 * cube_size + 2 * cube_spacing_y) / 2 + cube_size / 2
            for row in range(3):
                for col in range(3):
                    cube_x = x_start + col * (cube_size + cube_spacing_x)
                    cube_y = y_start + row * (cube_size + cube_spacing_y)
                    cube_z = z_pos + board_height/2 + cube_size/2
                    cube_pose = SE3(cube_x, cube_y, cube_z)
                    self.cube_poses.append(cube_pose)
                    cube = Cuboid(scale=[cube_size, cube_size, cube_size], color=cube_colors[color_key], pose=SE3(cube_pose))
                    self.env.add(cube)
                    self.ingredient_objects.append(cube)
                    self.cube_objects.append(cube)

        print("Scene configuration loaded.")