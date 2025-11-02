import swift
from collision_checker import CollisionChecker
from EnvironmentSetuptest import Scene
from GlassBot import GlassBot
from DrinkBot import DrinkBot
from IngredientBot import IngredientBot
from ServerBot import ServerBot
import numpy as np

def setup_collision_checker(env, scene):
    checker = CollisionChecker(env=env, visualise=True)
    # Register all static cuboid-like objects (walls, tables, mats, bases, etc.)
    for obj in getattr(scene, "static_objects", []):
        checker.add_scene_prisms(obj)
    return checker

def main():
    env = swift.Swift()
    env.launch(realtime=True)

    scene = Scene(env)
    print("Scene configuration loaded.")

    # Instantiate robots
    drinkbot = DrinkBot()
    ingredientbot = IngredientBot()
    glassbot = GlassBot()
    serverbot = ServerBot()

    glassbot.base = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"]
    drinkbot.base = scene.ROBOT_BASE_POSES["R2_ALCOHOL"]
    ingredientbot.base = scene.ROBOT_BASE_POSES["R3_MIXERS"]
    serverbot.base = scene.ROBOT_BASE_POSES["R4_SERVER"]

    # Add robots to swift env
    for bot in [drinkbot, ingredientbot, glassbot, serverbot]:
        bot.add_to_env(env)

    # Setup collision checker with scene objects (tables, walls, etc.)
    checker = setup_collision_checker(env, scene)

    print("--- Collision Check at Home Positions ---")
    for bot in [drinkbot, ingredientbot, glassbot, serverbot]:
        pts = checker.check_collision_for_q(bot, bot.q, return_all=True)

        if pts:
            print(f"{bot.name}: Collision detected! (found {len(pts)} point(s))")
        else:
            print(f"{bot.name}: No collisions detected")

    # Optional: create sliders to sweep joints interactively and re-check collisions
    def create_sliders_for_robot(robot, sim_env, collision_checker):
        sliders = []
        def slider_cb(value, joint_idx):
            new_q = robot.q.copy()
            new_q[joint_idx] = np.deg2rad(float(value))
            robot.q = new_q

            coll = collision_checker.check_collision_for_q(robot, robot.q)
            if coll:
                print(f"{robot.name}: Collion detected at q={np.round(np.rad2deg(robot.q),2)}!")
            else:
                print(f"{robot.name}: No collisions detected at q={np.round(np.rad2deg(robot.q),2)}")

        for i in range(getattr(robot, 'n', len(robot.q))):
            qmin = np.rad2deg(robot.qlim[0, i]) if hasattr(robot, 'qlim') else -180
            qmax = np.rad2deg(robot.qlim[1, i]) if hasattr(robot, 'qlim') else 180
            init = np.rad2deg(robot.q[i]) if hasattr(robot, 'q') else 0
            s = swift.Slider(cb=lambda v, j=i: slider_cb(v, j),
                             min=qmin, max=qmax, step=1, value=init,
                             desc=f'Joint {i+1} Angle', unit='°')
            sliders.append(s)

        for s in sliders:
            sim_env.add(s)

        print("Sliders added. Use the Swift window to manipulate joints. Press Ctrl+C in terminal to exit teach mode.")
        try:
            while True:
                sim_env.step(0.02)
        except KeyboardInterrupt:
            print("Exiting teach mode...")

        for s in sliders:
            sim_env.remove(s)

    choice = input("Enter robot to teach (1=DrinkBot,2=IngredientBot,3=GlassBot,4=ServerBot) or press Enter to skip: ")
    mapping = {'1': drinkbot, '2': ingredientbot, '3': glassbot, '4': serverbot}
    if choice in mapping:
        create_sliders_for_robot(mapping[choice], env, checker)

    input("Press Enter to close Swift...")
    env.close()

if __name__ == "__main__":
    main()
