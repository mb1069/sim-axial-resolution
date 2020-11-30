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

end