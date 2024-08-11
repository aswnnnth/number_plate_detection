import cv2
import os
import pytesseract
import pandas as pd

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load the Haar Cascade for license plate detection
cascade_path = r'C:\Users\aswanth k\OneDrive\Documents\Dl\number_plate_dtection\haarcascade_russian_plate_number.xml'
harcascade = cv2.CascadeClassifier(cascade_path)

# Check if the Haar Cascade is loaded correctly
if harcascade.empty():
    print("Error loading Haar Cascade.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Define the minimum area of the detected object
min_area = 500
count = 0

# Ensure the directory exists for saving images
output_dir = 'plates'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to Tesseract executable (adjust this to your installation path)
pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe')

# List to store recognized license plate numbers
plate_numbers = []

while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale
    gray_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect license plates
    plates = harcascade.detectMultiScale(gray_cap, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Draw rectangle around detected plate
            cv2.rectangle(frame, pt1=(x, y),pt2= (x + w, y + h),color= (0, 255, 0),thickness= 3)
            cv2.putText(frame, 'Number Plate', (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 3)

            # Extract Region of Interest (ROI)
            frame_roi = frame[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            thresh,bw= cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
            cv2.imshow('Plate ROI', gray_roi)

    
    # Display the resulting frame
    cv2.imshow("Result", frame)

    # Save the plate image when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if 'frame_roi' in locals():   # Check if frame_roi exists
            filename = os.path.join(output_dir, "scanned_img" + str(count) + ".jpg")
            cv2.imwrite(filename,gray_roi)
            cv2.rectangle(frame, pt1=(0, 200), pt2=(640, 300), color=(0, 255, 0), thickness=cv2.FILLED)
            cv2.putText(frame, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Result", frame)
            cv2.waitKey(500)
            
            # Extract text from the saved image using Tesseract
            extracted_text = pytesseract.image_to_string(filename,config=r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            print(f"Extracted Number: {extracted_text.strip()}")

            # Append the recognized text to the list
            plate_numbers.append({"Image": filename, "Plate Number": extracted_text.strip()})

            count += 1  # Increment the count

    # Exit the loop when 'q' is pressed
    elif key == ord('q'):
        break

# Save the extracted license plate numbers to an Excel file
df = pd.DataFrame(plate_numbers)
df.to_excel('recognized_plates.xlsx', index=False)
print("License plate numbers saved to recognized_plates.xlsx")

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

