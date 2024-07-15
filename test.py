import cv2
import numpy as np


def remove_ruido(thresh):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    min_size = 80  # Tamanho mínimo de componente para manter
    cleaned_image = np.zeros(thresh.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = 255
    return cleaned_image

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        grayFrame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(grayFrame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        img = remove_ruido(thresh)
        cv2.imshow("Camera", img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()