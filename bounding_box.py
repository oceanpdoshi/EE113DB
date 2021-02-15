import cv2
import matplotlib.pyplot as plt

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def draw_bbox(image_path, bboxes, scores=[]):
    imgcv = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    # opencv assumes BGR image rather than RGB -> flip channels
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
    if scores == []:
        for cat in bboxes.keys():
            for box in bboxes[cat]:
                x, y, w, h = box
                x1 = int(x)
                y1 = int(y)
                x2 = int(x+w)
                y2 = int(y+h)
                cv2.rectangle(imgcv, (x1,y1), (x2,y2), (0,255,0), 2) # add rectangle to image
                
                label = cat
                labelSize=cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN,0.5,2)
                _x1 = x1
                _y1 = y1#+int(labelSize[0][1]/2)
                _x2 = _x1+labelSize[0][0]
                _y2 = y1-int(labelSize[0][1])
                cv2.rectangle(imgcv,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
                cv2.putText(imgcv,label,(x1,y1),cv2.FONT_HERSHEY_PLAIN,0.5,(255,255,255),1)
                
                plt.imshow(imgcv)
    else:
        # TODO - write code to also plot scores (in addition to labels)
        print(0)
    plt.show()
    