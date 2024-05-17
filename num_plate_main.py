import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import cvzone
import math
# import PIL.Image
import enum
import pytesseract as pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class detect:
    
    def __init__(self,impath,mode):
        self.impath = impath
        self.mode = mode

    def findnumplate(impath,mode):

        image = cv2.imread(impath)
        cv2.imshow("preview_Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        machine=r"C:\My_Things\apps\vs_code\virtual_environments\yolo_models\num_plate_v1.pt"
        model= YOLO(machine)
        results=model(image, stream=True)
        # global cpimg,croppedimg,gray

        for r in results:
                boxes=r.boxes
                for box in boxes:
                    # global cpimg
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1=int(x1),int(y1)
                    bbox=int(x1),int(y1),int(x2-x1),int(y2-y1)
                    cpimg=image.copy()
                    cvzone.cornerRect(cpimg,bbox) 
                    conf=math.ceil((box.conf[0]*100))/100
                    cvzone.putTextRect(cpimg,f'{conf}',(x1,y1),scale=1,thickness=1)

        cv2.imshow("bounding_box",cpimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        x,y,w,h=bbox
        croppedimg=image[y:y+h,x:x+w]
        cv2.imshow("cropped_img",croppedimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        gray = cv2.cvtColor(croppedimg, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray,9,75,75)
        
        if mode == 0:

            print("using pytesseract for character recognition ")
            myconfig = r"--psm 12 --oem 3"
            text=pytesseract.image_to_string(gray, config=myconfig)
            print("predicted car number plate value : ",text)

        elif mode == 1:

            print("using YOLO machine for character recognition ")
            machine=r"C:\My_Things\apps\vs_code\virtual_environments\yolo_models\numplate_char_v1.pt"
            model= YOLO(machine)

            # gray = cv2.resize(gray, (960, 540))
            # cv2.imshow("number_plate", croppedimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            croppedimg = cv2.bilateralFilter(croppedimg,9,75,75)
            results=model(croppedimg)

            classNames = ['0', '1', '10', '2', '3', '4', '5', '6', '66', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'M-', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'm']

            clsnm = []
            coords = []
            for r in results:
                    boxes=r.boxes
                    for box in boxes:
                        x1,y1,x2,y2=box.xyxy[0]
                        x1,y1=int(x1),int(y1)
                        x2,y2=int(x2),int(y2)
                        coords.append((x1,y1))
                        cls=int(box.cls[0])
                        clsnm.append(classNames[cls])
                        cv2.rectangle(croppedimg,(x1,y1),(x2,y2),(255,0,0),1)
                        cvzone.putTextRect(croppedimg,f'{classNames[cls]}',(x1,y1),scale=1,thickness=1)

            print(clsnm)
            cv2.imshow("img",croppedimg)
            cv2.waitKey(0)

            zlist=zip(coords,clsnm)
            
            zlist = list(zlist)
            zlist=sorted(zlist,key = lambda x:x[0][0])
            print(zlist)

            chars = []
            for element in zlist:

                f1 = element[1]
                chars.append(f1)

            print(chars)
            plate_chars = ' '.join(chars)
            print(plate_chars)

    def controls(mode):
         
        if mode == 0:
             print("use pytesseract for character recognition ")
        elif mode == 1:
             print("use YOLO machine for character recognition ")