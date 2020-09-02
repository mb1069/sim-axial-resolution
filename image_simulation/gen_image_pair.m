% Wrapped function to generate input/output training data from a point
% cloud

% Usage: 
% to generate a random point cloud in a sphere;
% gen_image_pair(<SIM input resolution>, -1, <number of points in sphere>, 2048, 'in.tif', 'out.tif')

% to generate a chromatin structure 
% gen_image_pair(<SIM input resolution>, <structure index (1-100), -1, 2048, 'in.tif', 'out.tif')


function points = gen_image_pair(N, dataIndex, npoints, nphot, imgInName, imgOutName)
    if dataIndex == -1
        points = load_point_cloud(npoints);
    else
        points = load_data_points(dataIndex);
    end
    showfigures = false;
    gen_expanded_stack;

    'Saving images'
    in8 = uint8(img/max(img(:))*255);
    imstackwrite(in8, imgInName);
    
    output_img = gen_output_stack(N*2, points);
    sim_image = uint8(output_img/max(output_img(:))*255);
    imstackwrite(sim_image, imgOutName);

