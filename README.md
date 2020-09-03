# Improving axial resolution in SIM using deep learning

Miguel Boland <sup>1</sup>, Dr. Edward A.K. Cohen <sup>1</sup>, Dr. Seth Flaxman<sup>1</sup> & Professor Mark Neil <sup>2</sup>
 
<sup>1</sup>Imperial College London, Faculty of Natural Sciences, Department of Mathematics
<sup>2</sup>Imperial College London, Faculty of Natural Sciences, Department of Physics

Imperial College London

## Repository organisation
The code in this repository is organised as follows:

### Image simulation
Contains all code required to generate the training data used in our experiment, in both raw SIM image stacks (inputs) and a high resolution confocal output image (target)
### Deep learning
Contains all code required to generate training data from existing TIFF files, and produce a trained RCAN model on this data.

### Image evaluation
Contains code required to run the RCAN models on a set of test images, and generate comparison metrics with SIM reconstructions and simulated high-resolution target data.

## Please feel free to ask any questions using Github Issues!