"""
test.py
Uses the simplified CollisionChecker (no adapters). Adds robots + scene and checks collisions.
"""

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

    scene = Scene(env)   # Scene populates env and scene.static_objects
    print("Scene configuration loaded.")

    # instantiate robots
    drinkbot = DrinkBot()
    ingredientbot = IngredientBot()
    glassbot = GlassBot()
    serverbot = ServerBot()

    # optionally set robot base poses from scene if robots support .base
    try:
        glassbot.base = scene.ROBOT_BASE_POSES["R1_ICE_GLASS"]
    except Exception:
        pass
    try:
        drinkbot.base = scene.ROBOT_BASE_POSES["R2_ALCOHOL"]
    except Exception:
        pass
    try:
        ingredientbot.base = scene.ROBOT_BASE_POSES["R3_MIXERS"]
    except Exception:
        pass
    try:
        serverbot.base = scene.ROBOT_BASE_POSES["R4_SERVER"]
    except Exception:
        pass

    # optionally set home q
    for bot in [drinkbot, ingredientbot, glassbot, serverbot]:
        if hasattr(bot, "home_q"):
            try:
                bot.q = bot.home_q
            except Exception:
                pass

    # add robots to swift env (try .add_to_env, fallback to env.add)
    for bot in [drinkbot, ingredientbot, glassbot, serverbot]:
        try:
            bot.add_to_env(env)
        except Exception:
            try:
                env.add(bot)
            except Exception:
                print(f"Failed to add {getattr(bot, 'name', str(bot))} to env")

    # Setup collision checker with static objects (tables, walls, etc.)
    checker = setup_collision_checker(env, scene)

    print("--- Collision Check at Home Positions ---")
    for bot in [drinkbot, ingredientbot, glassbot, serverbot]:
        try:
            pts = checker.check_collision_for_q(bot, bot.q, return_all=True)
        except Exception as e:
            print(f"{bot.name}: collision debug check error: {e}")
            pts = None
        if pts:
            print(f"{bot.name}: ❌ COLLISION detected (found {len(pts)} point(s))")
        else:
            print(f"{bot.name}: ✅ CLEAR")

    # Optional: create sliders to sweep joints interactively and re-check collisions
    def create_sliders_for_robot(robot, sim_env, collision_checker):
        sliders = []
        def slider_cb(value, joint_idx):
            try:
                new_q = robot.q.copy()
                new_q[joint_idx] = np.deg2rad(float(value))
                robot.q = new_q
            except Exception:
                return

            try:
                coll = collision_checker.check_collision_for_q(robot, robot.q)
                if coll:
                    print(f"{robot.name}: ❌ COLLISION at q={np.round(np.rad2deg(robot.q),2)}")
                else:
                    print(f"{robot.name}: ✅ CLEAR at q={np.round(np.rad2deg(robot.q),2)}")
            except Exception as e:
                print(f"Collision check error: {e}")

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
            try:
                sim_env.remove(s)
            except Exception:
                pass

    try:
        choice = input("Enter robot to teach (1=DrinkBot,2=IngredientBot,3=GlassBot,4=ServerBot) or press Enter to skip: ")
    except Exception:
        choice = ''
    mapping = {'1': drinkbot, '2': ingredientbot, '3': glassbot, '4': serverbot}
    if choice in mapping:
        create_sliders_for_robot(mapping[choice], env, checker)

    input("Press Enter to close Swift...")
    env.close()

if __name__ == "__main__":
    main()
