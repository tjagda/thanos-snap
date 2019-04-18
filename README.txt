****************************************************
Name: Removing objects from photos
Author: Timothy Agda
****************************************************

Description
***********
A blending algorithm that scans all the images in a 
folder, aligns them based on features, then blends 
all the images together using median blending. Resulting
in an image with only the static objects/scene.

How To Run
***********
Be sure to have Python3, Numpy and OpenCV installed. To 
choose a different folder to run the algorithm from, 
go into source code and change the 'dir' variable.

Then run the following command to run the algorithm 
and the percentage test:
    python median.py
    
NOTE: When median.py does the evaluation it assumes that the
'_grndT' folder is in the same folder as median.py
    
If you want to only run the percentage test, change the
filepath variables in the 'evaluate.py'

Then running the following:
    python evaluate.py
    
NOTE: This is a very slow algorithm on Python. 
Large pictures may take upwards to 20 min to complete.