# Scripts in this directory
* gen_image_pair.m generates an input SIM stack / output HR confocal image of either a chromatin structure (loaded from 50100.mat, itself generated from the original paper [Siyu Wang, Jinbo Xu, and Jianyang Zeng.
Inferential modeling of 3D chromatin structure.
Nucleic Acids Research, 43(8):54, 02 2015.]).
* fairsim.m runs the fairSIM reconstruction algorithm on a given input file.
* fairsim_dir.m runs fairsim.m on a directory of input images.
* gen_expanded_stack.m generates a SIM image stack according to the microscope setup described in our report.
* gen_output_stack.m generates an output high-res confocal image of a chromatin structure or point cloud (script is used by gen_point_pair.m).
* imstackread.m / imstackwrite.m are utility functions to read/write TIFF image stacks.
* load_data_points.m loads chromatin structures from 50100.mat (see above paper).
* load_point_cloud.m generates a random point cloud encased in a sphere.
* hexSimProcessor.m is a class implementing the SIM reconstruction algorithm used above for the hexSIM microscopy setup described in our report.