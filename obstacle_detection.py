import cv2
import numpy as np
import time

def detect_obstacles(video_path, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_times = []

    prev_color = None

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_frame = time.time()

        # Convert the frame to the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of colors for obstacle detection
        lower_lightred = np.array([0, 100, 20])
        upper_lightred = np.array([179, 255, 255])

        lower_lightgreen = np.array([40, 50, 50])
        upper_lightgreen = np.array([80, 255, 255])

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        # Create masks using color ranges
        mask_lightred = cv2.inRange(hsv, lower_lightred, upper_lightred)
        mask_lightgreen = cv2.inRange(hsv, lower_lightgreen, upper_lightgreen)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Combine masks
        mask = cv2.bitwise_or(mask_lightred, cv2.bitwise_or(mask_lightgreen, mask_white))

        # Find contours in the combined mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # Draw rectangles around the detected objects
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                shape = "Triangle"
            else:
                shape = "Circle"

            # Get the average color of the bounding box
            roi = frame[y:y + h, x:x + w]

            # Get the average color of the non-black pixels in the bounding box
            non_black_pixels = roi[np.any(roi != [0, 0, 0], axis=-1)]
            if len(non_black_pixels) > 0:
                avg_color = np.mean(non_black_pixels, axis=(0))

            else:
                # Handle the case where there are no non-black pixels
                avg_color = np.array([0, 0, 0])

            color_detected = None

            # Check the average color
            if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:  # Red channel is dominant
                color_detected = "Red"
            elif avg_color[1] > avg_color[2] and avg_color[1] > avg_color[0]:  # Green channel is dominant
                color_detected = "Green"
            elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:  # Blue channel is dominant
                color_detected = "Blue"

            # Add label to the detected object
            label = f"{color_detected} {shape}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw rectangles around the detected objects
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the result
        #cv2.imshow('Obstacle Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        end_frame = time.time()
        frame_times.append(end_frame - start_frame)

        # Write the frame to the output video
        video_writer.write(frame)

    finish_time = time.time()

    # Add a delay of 1 seconds (1000 milliseconds)
    cv2.waitKey(1000)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    cap.release()
    video_writer.release()

    total_detection_time = finish_time - start_time
    print(f"Total time for all objects to be detected: {total_detection_time} seconds")
    # Print the average time per frame
    average_time_per_frame = sum(frame_times) / len(frame_times)
    print(f"Average time per frame: {average_time_per_frame} seconds")


if __name__ == "__main__":
    video_path = 'test_video.mp4'
    detect_obstacles(video_path)
