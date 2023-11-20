import cv2
import numpy as np
import time

def is_overlapping(position, drawn_positions, min_distance):
    for other_position in drawn_positions:
        distance = np.linalg.norm(np.array(position) - np.array(other_position))
        if distance < min_distance:
            return True
    return False

def generate_video(video_path, width=640, height=480, fps=30, seconds=10, max_shape_size=40):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    prev_color = None

    # Set a fixed seed for reproducibility
    np.random.seed(35)
    available_colors = ['red', 'green', 'blue']

    for _ in range(seconds):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        drawn_positions = set()

        for _ in range(9):
            while True:
                x, y = np.random.randint(50, width - 50), np.random.randint(50, height - 50)
                if not is_overlapping((x, y), drawn_positions, 70):
                    drawn_positions.add((x, y))
                    break

            color = np.random.choice(available_colors)

            if color == 'red':
                rgb_color = (0, 0, 255)  # Red
            elif color == 'green':
                rgb_color = (0, 255, 0)  # Green
            else:
                rgb_color = (255, 0, 0)  # Blue

            # Randomly choose a shape to draw: 0 for circle, 1 for rectangle, 2 for triangle
            shape_choice = np.random.randint(0, 2)

            if shape_choice == 0:
                # Draw a circle
                #radius = max_shape_size
                cv2.circle(frame, (x, y), 20, rgb_color, -1)
            else:
                # Draw a triangle
                vertices = np.array([[x, y - np.random.randint(5, max_shape_size)],
                                     [x - np.random.randint(5, max_shape_size),
                                      y + np.random.randint(5, max_shape_size)],
                                     [x + np.random.randint(5, max_shape_size),
                                      y + np.random.randint(5, max_shape_size)]],
                                    np.int32)
                vertices = vertices.reshape((-1, 1, 2))
                cv2.fillPoly(frame, [vertices], rgb_color)
        for _ in range(fps):
            #cv2.circle(frame, (x, y), 20, rgb_color, -1)
            video_writer.write(frame)


    video_writer.release()


if __name__ == "__main__":
    video_path = 'test_video.mp4'
    generate_video(video_path)
