function [tiff_stack] = imstackread(filename)
    tiff_info = imfinfo(filename); % return tiff structure, one element per image
    tiff_stack = imread(filename, 1) ; % read in first image
    %concatenate each successive tiff to tiff_stack
    for ii = 2 : size(tiff_info, 1)
        temp_tiff = imread(filename, ii);
        tiff_stack = cat(3 , tiff_stack, temp_tiff);
    end
end