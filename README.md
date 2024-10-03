# Mesh-Manipulation-and-Coating-Transfer

This is the work I have made as part of my computer vision project for the corresponding semester course. It is a mesh editor that uses various techniques to achieve the desirable effects. 

Those techniques include Taubin smoothing (global and local), skeletization of the model using Laplacian smoothing (global and local), skeletization using the theory of the following paper https://cgv.cs.nthu.edu.tw/projects/Shape_Analysis/skeleton for more accurate results, increase and decrease in resolution of the model and coat extraction and application to the same or a different model.

After executing the GUI.py a file explorer window allows you to open the model to be edited.

Keyboard Commands Manual
- A applies taubin smoothing globally.
- B , after selecting a point with CTRL + mouse click, applies taubin smoothing locally.
- C applies skeletization globally.
- D , after selecting a point with CTRL + mouse click, applies skeletization locally .
- E samples the model so the skeletization or smoothing can be done faster
- F decreases the resolution of the model.
- G increases the resolution of the model.
- H , after selecting a point with CTRL + mouse click, it extracts the local coating of the model. After that the same or a new model can be selected and with reselecting a point and pressing H again, the coating will be applied to the the model.

Requirements:
- Python 3.10.11
- matplotlib 3.7.1
- numpy 1.24.2
- scipy 1.11.1
- Open3D 0.16.0
