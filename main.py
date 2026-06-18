import pyray as pr


def main():
    pr.init_window(800, 450, "Hello Pyray")
    pr.set_target_fps(60)

    while not pr.window_should_close():
        pr.begin_drawing()
        pr.clear_background(pr.RAYWHITE)
        pr.end_drawing()

    pr.close_window()


if __name__ == "__main__":
    main()
