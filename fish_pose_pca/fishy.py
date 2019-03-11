import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math
import copy

class FishBlob:
    def __init__(self, X, Y, centroid):
        #X and Y represents a contour
        self.X = X
        self.Y = Y
        self.length = len(X)
        self.centroid = centroid
        
    #Negative Index shift means "rotating the array forward"
    def shiftContourStart(self, indexShift):
        cut = indexShift%self.length
        self.X, self.Y = np.concatenate((self.X[cut:],self.X[:cut]),axis=0), np.concatenate((self.Y[cut:],self.Y[:cut]),axis=0)

    def translate(self, vector):
        M1 = np.float32([[1,0,vector[0]],[0,1,vector[1]]])
        M2 = np.float32([self.X,self.Y,np.ones((self.length))])
        self.centroid = self.centroid + vector
        self.X, self.Y = translated = M1@M2
        
    def rotate(self, angleInDegrees):
        #The last parameter is a scaling factor. Setting it to one disables the effect (multiplication by 1)
        M1 = cv2.getRotationMatrix2D((self.centroid[0],self.centroid[1]), angleInDegrees, 1)
        M2 = np.float32([self.X,self.Y,np.ones((self.length))])
        self.X, self.Y = rotated = M1@M2

        
    def getDistancesFromCentroid(self):
        diff_X,diff_Y = self.X-self.centroid[0], self.Y-self.centroid[1]
        return np.linalg.norm(np.array([diff_X,diff_Y]),axis=0)
    
    #Returns the index of the point in the Blob, that is the farthest away from the center of mass
    def argFurthestPointFromCentroid(self):
        distances = self.getDistancesFromCentroid()
        return np.argmax(distances)
    
    #From the point of view of the fish's vector [1,0]
    #Positive angle means clockiwise rotated from [1,0], negative angle counter-clockwise
    def getAngleOfPoint(self, point):
        x, y = point - self.centroid
        return math.atan2(y,x)
    
    def reduceToSubfish(self,newSize):
        selector = sorted(random.sample(range(0,self.length),newSize))
        self.X, self.Y = self.X[selector],self.Y[selector]
        self.length = newSize
    
    
def randomSubfishes(fishes, subarrayLen):
    ret = np.empty((len(fishes),2,subarrayLen))
    for i,fish in enumerate(fishes):
        selector =sorted(random.sample(range(0,len(fish[0])),subarrayLen))
        ret[i][0],ret[i][1] = fish[0][selector],fish[1][selector]
    return ret

def normalizeFish(X,Y,c):
    nFish = FishBlob(X=X,Y=Y,centroid=c)
    
    #Normalize Contour Start
    tailIndex = nFish.argFurthestPointFromCentroid()
    nFish.shiftContourStart(tailIndex)
    
    #Normalize Position
    nFish.translate(vector=-1*nFish.centroid)
    
    #Normalize Rotation
    nFish.rotate(math.degrees(nFish.getAngleOfPoint(np.array([nFish.X[0],nFish.Y[0]]))))

    return nFish
           
    
def printFish(fishBlob,canvasSize=200):
    blob = copy.copy(fishBlob)
    blob.translate(np.array([canvasSize//2,canvasSize//2]))
    X,Y = blob.X, blob.Y
    dist = blob.getDistancesFromCentroid()
    canvas=np.zeros((canvasSize,canvasSize))
    canvas[Y[:20].astype(int),X[:20].astype(int)] = 140
    canvas[Y[20:].astype(int),X[20:].astype(int)] = 255
    #canvas[Y.astype(int),X.astype(int)] = dist
    #canvas[fishBlob.centroid[0]-3:fishBlob.centroid[0]+3,fishBlob.centroid[1]-3:fishBlob.centroid[1]+3] = 90
    plt.figure(figsize=(15,15))
    plt.imshow(canvas,cmap="gray")
    plt.show()