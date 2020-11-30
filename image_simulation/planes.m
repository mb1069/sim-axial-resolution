function [points] = sample_plane(z, npoints)
    start=-1;
    f=1;
    
    rand_xy = (f-start).*rand(npoints, 2) + start;
    z = transpose(repelem(z, npoints));
    points = [rand_xy, z]
end
