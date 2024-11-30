import numpy as np
import cv2 as cv
import math
import logging
import os
from pathlib import Path

from synthetic_dataset import generate_background
from synthetic_dataset import draw_lines
from synthetic_dataset import draw_polygon
from synthetic_dataset import draw_multiple_polygons
from synthetic_dataset import draw_ellipses
from synthetic_dataset import draw_star
from synthetic_dataset import draw_checkerboard
from synthetic_dataset import draw_stripes
from synthetic_dataset import draw_cube
from synthetic_dataset import gaussian_noise  

BASE_PATH = "synthetic_shapes_v17"
basePath  = Path(BASE_PATH)
if not os.path.exists(basePath):
    print("[INFO] creating '{}' directory".format(basePath))
    os.makedirs(basePath)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)-5s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

'''
datasets = [
            "draw_lines",
            "draw_polygon",
            "draw_multiple_polygons",
            "draw_ellipses",
            "draw_star",
            "draw_checkerboard",
            "draw_stripes",
            "draw_cube",
            "gaussian_noise"
            ]
'''

datasets = ["gaussian_noise"]      

'''
datasets = [
            "draw_lines",
            "draw_polygon",
            "draw_multiple_polygons",
            "draw_ellipses",
            "draw_star",
            "draw_checkerboard",
            "draw_stripes",
            "draw_cube"]
'''

    

actions = [("test", 500), ("training", 10000), ("validation", 200)]
holders = ["images", "points"]

for dataset in datasets:
    # create primitive directory
    primitivePath = basePath / dataset       # datasets/draw_lines
    if not os.path.exists(primitivePath):
        print("[INFO] creating '{}' directory".format(primitivePath))
        os.makedirs(primitivePath, exist_ok=True)
   
    holdersPath = []
    for holder in holders:                   # datasets/draw_lines/images
       holderPath = primitivePath / holder   # datasets/draw_lines/points
       holdersPath.append(holderPath)
       if not os.path.exists(holderPath):
           print("[INFO] creating '{}' directory".format(holderPath))
           os.makedirs(holderPath, exist_ok=True)
    
    for (action, fileNumber) in actions:     
        actionsPath = []
        for i in range(len(holdersPath)):     
            actionPath = holdersPath[i] / action  # datasets/draw_lines/images/test
            actionsPath.append(actionPath)        # datasets/draw_lines/points/test
            if not os.path.exists(actionPath):
                print("[INFO] creating '{}' directory".format(actionPath))
                os.makedirs(actionPath, exist_ok=True)
        
        for i in range(fileNumber):
            img     = generate_background()
            points  = eval(dataset)(img)
            imgPath = str(i) + ".png" 
            ptsPath = str(i) + ".npy"
            imgPath = actionsPath[0] / imgPath  # datasets/draw_lines/images/test/0.png
            ptsPath = actionsPath[1] / ptsPath  # datasets/draw_lines/points/test/0.npy
            
            # write image and corner points into files
            '''
            np.save(str(ptsPath), points)
            cv.imwrite(str(imgPath), img)
            '''
            save_file = "export3.txt"            
            save_file2 = "export4.txt"            
            
            if points.any()== 0:           
                with open(save_file, "a") as myfile:    # 1
                    myfile.write(str(imgPath) + "\n")
                    myfile.write(str(ptsPath) + "\n")      
                    myfile.write(str(points)  + "\n")      
                    
                    img     = generate_background()
                    points  = eval(dataset)(img)                    
                    np.save(str(ptsPath), points)
                    cv.imwrite(str(imgPath), img)    
                    
                    np.save(str(ptsPath), points)
                    cv.imwrite(str(imgPath), img) 
                    
                    myfile.write(str(imgPath) + "\n")
                    myfile.write(str(ptsPath) + "\n")      
                    myfile.write(str(points)  + "\n")                     
                    
                    if points.any()== 0:              # 2
                        img = generate_background()
                        points = eval(dataset)(img)
                        
                        imgPath = str(i) + ".png" 
                        ptsPath = str(i) + ".npy"
                        imgPath = actionsPath[0] / imgPath  # datasets/draw_lines/images/test/0.png
                        ptsPath = actionsPath[1] / ptsPath  # datasets/draw_lines/points/test/0.npy  
                        
                        np.save(str(ptsPath), points)
                        cv.imwrite(str(imgPath), img)                         
                        
                        myfile.write(str(imgPath) + "\n")
                        myfile.write(str(ptsPath) + "\n")      
                        myfile.write(str(points)  + "\n")                 

                        if points.any()== 0:        # 3
                            img = generate_background()
                            points = eval(dataset)(img)
                            
                            imgPath = str(i) + ".png" 
                            ptsPath = str(i) + ".npy"
                            imgPath = actionsPath[0] / imgPath  # datasets/draw_lines/images/test/0.png
                            ptsPath = actionsPath[1] / ptsPath  # datasets/draw_lines/points/test/0.npy  
                            
                            np.save(str(ptsPath), points)
                            cv.imwrite(str(imgPath), img)                         
                            
                            myfile.write(str(imgPath) + "\n")
                            myfile.write(str(ptsPath) + "\n")      
                            myfile.write(str(points)  + "\n")                

                            if points.any()== 0:    # 4
                                img = generate_background()
                                points = eval(dataset)(img)
                                
                                imgPath = str(i) + ".png" 
                                ptsPath = str(i) + ".npy"
                                imgPath = actionsPath[0] / imgPath  # datasets/draw_lines/images/test/0.png
                                ptsPath = actionsPath[1] / ptsPath  # datasets/draw_lines/points/test/0.npy  
                                
                                np.save(str(ptsPath), points)
                                cv.imwrite(str(imgPath), img)                         
                                
                                myfile.write(str(imgPath) + "\n")
                                myfile.write(str(ptsPath) + "\n")      
                                myfile.write(str(points)  + "\n")             
                            
                                if points.any()== 0:    # 5
                                    img = generate_background()
                                    points = eval(dataset)(img)
                                    
                                    imgPath = str(i) + ".png" 
                                    ptsPath = str(i) + ".npy"
                                    imgPath = actionsPath[0] / imgPath  # datasets/draw_lines/images/test/0.png
                                    ptsPath = actionsPath[1] / ptsPath  # datasets/draw_lines/points/test/0.npy  
                                    
                                    np.save(str(ptsPath), points)
                                    cv.imwrite(str(imgPath), img)                         
                                    
                                    myfile.write(str(imgPath) + "\n")
                                    myfile.write(str(ptsPath) + "\n")      
                                    myfile.write(str(points)  + "\n")                                                 
                                pass                
            
            else:                
                with open(save_file2, "a") as myfile:
                    np.save(str(ptsPath), points)
                    cv.imwrite(str(imgPath), img)                                                 
                    myfile.write(str(imgPath) + "\n")
                    myfile.write(str(ptsPath) + "\n")                   
                    myfile.write(str(points)  + "\n")   
                pass  
                
                
                    
            
   




