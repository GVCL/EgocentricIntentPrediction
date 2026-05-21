import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' or 'Esc' to close the window.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame.")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Esc key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
