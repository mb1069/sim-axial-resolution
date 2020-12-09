
% Helper script to run fairsim for every input image in a directory
% Expected image setup is 3 angle * 2 phases, i.e 7 frames per
% reconstructed layer
function [] = fairsim_dir(dirname)
    input_images = dir(dirname+"*_in*.tif");
    input_image_names = {input_images(:).name};
    for f = 1 : length(input_image_names)
        in_img = string(dirname) + input_image_names(f);
        output_img = strrep(in_img, '_in', '_sim');
        if ~isfile(output_img)
            fairsim(in_img, output_img);
        end
    end
end