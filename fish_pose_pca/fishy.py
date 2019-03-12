"""object representation of a fish blob and surrounding utlity."""

from load_contours import load_contours
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
        
    def normalize(self):
        #Normalize Contour Start
        tailIndex = self.argFurthestPointFromCentroid()
        self.shiftContourStart(tailIndex)
        
        #Normalize Position
        self.translate(vector=-1*self.centroid)
        
        #Normalize Rotation
        self.rotate(math.degrees(self.getAngleOfPoint(np.array([self.X[0],self.Y[0]]))))
           
    
def get_fish_on_canvas(fish_blob=None, xy_representation=None, canvas_size=(200, 200)):
    """
    canvasSize = (y size, x size)

    """
    canvas=np.zeros(canvas_size)
    
    if fish_blob is not None:
        blob = copy.copy(fish_blob)
        blob.translate(np.array([canvas_size[1]//2,canvas_size[0]//2]))
        X,Y = blob.X, blob.Y
        canvas[Y.astype(int),X.astype(int)] = 255
    elif xy_representation is not None:
        X,Y = xy_representation + np.array([[canvas_size[1] // 2], [canvas_size[0] // 2]])
        canvas[Y.astype(int),X.astype(int)] = 255
    return canvas

def get_normalized_fish_blobs_from_data(data_path, mask_path):
    data = load_contours(path=data_path,mask_path=mask_path)
    allBlobs = []
    for d in data:
        fish = FishBlob(np.array(d.xs),np.array(d.ys),np.array([d.c_x,d.c_y]))
        fish.normalize()
        allBlobs.append(fish)
    return allBlobs