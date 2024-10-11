import cv2

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "data/input/video/Driver_zooms_through_red_light_plows_into_two_cars.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    print(success)
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # Save the annotated frame to a file
        frame_count += 1
        # cv2.imwrite(f"frame_{frame_count}.jpg", annotated_frame)
        # print(f"Saved frame {frame_count}")
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()