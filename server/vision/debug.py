import time
import cv2

def show_image(image):

    while True:
        # Display result
        cv2.imshow("frame", image)

        # time.sleep(0.2)
        break
        
        k = cv2.waitKey(1) & 0xff
        if k == 27: break # ESC pressed
            
    cv2.destroyAllWindows()