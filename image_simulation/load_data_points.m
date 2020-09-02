function [points] = load_data_points(dataIndex)
    scale = 10;
    disp("Loading realistic simulated data");
    try
        load('/rdsgpfs/general/user/mdb119/home/sim_project/HexSimulator/50100.mat');
    catch exception
        load('./50100.mat')
    end
    points = squeeze(Ensemble(dataIndex,:,:));

%   Rescaling points
    points = points/(max(max(abs(points)))/scale);

    data_shape = size(points);

%     TODO split into seperate function
%     if exist('dataIndex', 'var')
% 
%     else
%         %% Set up some random points
%         disp("Calculating point cloud");
%         rad = 5;        % radius of sphere/structure of points
%         tic
%         pointsx2 = (2*rand(npoints*2,3)-1).*[rad,rad,rad];
% 
%         pointsx2r = sum(pointsx2.*pointsx2,2);
% 
%         points_sphere = pointsx2(pointsx2r<rad^2,:);
% 
%         points = points_sphere((1:npoints),:);
%         points(:,3)=points(:,3)/2;
%         toc
%     end
end