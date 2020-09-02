function [points] = load_point_cloud(npoints)

    %% Set up some random points (sphere)
    disp("Calculating point cloud");
    rad = 5;        % radius of sphere/structure of points
    tic
    pointsx2 = (2*rand(npoints*2,3)-1).*[rad,rad,rad];

    pointsx2r = sum(pointsx2.*pointsx2,2);

    points_sphere = pointsx2(pointsx2r<rad^2,:);

    points = points_sphere((1:npoints),:);
    points(:,3)=points(:,3)/2;
    toc
    %% Set up some random points (square)
%     points = rand(npoints, 3);
%     points = points - [0.5, 0.5, 0.5]; % Centering
%     x_range = 15;
%     y_range = 15;
%     z_range = 0;
%     
%     points(:,1) = points(:,1) * x_range * 2;
%     points(:,2) = points(:,2) * y_range * 2;
%     points(:,3) = points(:,3) * z_range * 2;
%     
%     %% Random points (circle)
%     angles = [1:npoints]/npoints * pi*2;
%     radius = 6;
%     points = zeros(npoints, 3);
%     points(:,1) = radius * cos(angles);
%     points(:,2) = radius * sin(angles);
%     points(:,3) = 0;

end