import os
from PIL import Image, ImageDraw
import numpy as np


def mapping():
    mat = np.loadtxt('./created_data/hand_data/input_coordinate.txt', delimiter=',')

    npXZ = mat[:, [True, False, True]]
    INF = 1e9

    # (INF, INF, INF)ごとに処理を行う
    char_nplist = []
    char_pre_point = 0
    char_last_point = len(npXZ)

    for i, data in enumerate(npXZ):
        if data[0] == INF:
            char_nplist.append(npXZ[char_pre_point:i])
            char_pre_point = i+1
    if npXZ[char_last_point-1, 0] != INF:
        char_nplist.append(npXZ[char_pre_point::])

    most_left_point = 20.0
    most_lower_point = 240.0
    trim_left = 0.0
    trim_right = 0.0
    trim_upper = INF
    trim_lower = 0.0
    img = Image.new("RGB", (720, 360), "White")
    draw = ImageDraw.Draw(img)
    cut_level = 20.0

    for char_np in char_nplist:
        min_cx = INF
        max_cz = -INF
        char_list = char_np.tolist()

        for char_data in char_list:
            min_cx = min(min_cx, char_data[0])
            max_cz = max(max_cz, char_data[1])

        adjust_x = most_left_point - min_cx
        adjust_z = most_lower_point - max_cz

        for char_data in char_list:
            char_data[0] = char_data[0] + adjust_x
            char_data[1] = char_data[1] + adjust_z
            most_left_point = max(most_left_point, char_data[0])
            trim_right = max(trim_right, char_data[0])
            trim_upper = min(trim_upper, char_data[1])
            trim_lower = max(trim_lower, char_data[1])

        most_left_point = most_left_point + cut_level
        char_tuple = tuple(map(tuple, char_list))
        pre_x = 0.0
        pre_z = 0.0
        pre_point = 0

        for i, xz_point in enumerate(char_list):
            if i != 0:
                pre_xz = np.array([pre_x, pre_z])
                now_xz = np.array([xz_point[0], xz_point[1]])
                distance_xz = np.linalg.norm(pre_xz - now_xz)
                if distance_xz > cut_level:
                    draw_tuple = char_tuple[pre_point:i]
                    draw.line(draw_tuple, fill="Black", width=5, joint="curve")
                    pre_point = i
            pre_x = xz_point[0]
            pre_z = xz_point[1]

        last_tuple = char_tuple[pre_point::]
        draw.line(last_tuple, fill="Black", width=5, joint="curve")

    img_margin = 20.0
    trim_img = img.crop((trim_left, trim_upper-img_margin,
                        trim_right+img_margin, trim_lower+img_margin))
    trim_img.save("./created_data/target_image/sample_trim.png")
    trim_img.save(f"./stored_image/sample_trim_{trim_upper}.png")
    os.remove("./created_data/hand_data/input_coordinate.txt")
    os.remove("./created_data/hand_data/finish.txt")
