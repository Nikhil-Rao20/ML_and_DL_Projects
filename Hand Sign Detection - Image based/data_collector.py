import cv2
import os

# Define the directory where you want to save the images
output_directory = "captured_images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras (0 is the default camera)

# Initialize the class counter
class_counter = 1
images_per_class = 100

# Define the ROI coordinates (left, top, width, height)
roi_x = 50
roi_y = 50
roi_width = 224
roi_height = 224

# Initialize the image counter for the current class
image_counter = 0

# Create a directory for the current class if it doesn't exist
class_directory = os.path.join(output_directory, str(class_counter))
if not os.path.exists(class_directory):
    os.makedirs(class_directory)

# Initialize variables for drawing the bounding box
bbox_color = (0, 255, 0)  # Green color
bbox_thickness = 2

# Create a window for display
cv2.namedWindow("Capture Images", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing the frame.")
        continue

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), bbox_color, bbox_thickness)

    # Display the frame
    cv2.imshow("Capture Images", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Check for spacebar press to capture an image
        # Crop the frame to the specified ROI
        roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Define the filename for the captured image
        filename = os.path.join(class_directory, f"image_{image_counter:03d}.jpg")

        # Save the ROI as an image
        cv2.imwrite(filename, roi)
        print(f'Saved: {filename}')

        image_counter += 1

        # If the desired number of images for the current class is reached, move to the next class
        if image_counter >= images_per_class:
            class_counter += 1
            image_counter = 0
            class_directory = os.path.join(output_directory, str(class_counter))
            if not os.path.exists(class_directory):
                os.makedirs(class_directory)

    if key == 27:  # Check for ESC key press to exit the loop
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
