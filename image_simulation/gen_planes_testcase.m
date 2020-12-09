function [] = gen_planes_testcase(spacing, imgInName, imgOutName)
    range = 0:0.0316:2;
    [m,n] = ndgrid(range, range);
    p1 = [m(:), n(:)];
    p1(:,3) = 0;
    for i = 1:4
        [m,n] = ndgrid(range, range);
        p2 = [m(:), n(:)];
        p2(:,3) = spacing * i;
        p1 = [p1;p2];
    end
    scatter3(p1(:,1), p1(:,2), p1(:,3))
    points = p1;
    size(points)


    % p1 = planes(0, 2000);
    % grid_spacing = 0.9;
    % for i = 1:2
    %     p2 = p1;
    %     p2(:,3) = (grid_spacing * i); % same xy, different z
    %     p1 = [p1; p2];
    % end
    % 
    % points = p1;
    % plot3(p1(:,1), p1(:,2), p1(:,3))
    % 
    N = 256;
    nphot = -1; % no noise

    showfigures = false;
    gen_expanded_stack;

    'Saving images'
    in8 = uint8(img/max(img(:))*255);
    imstackwrite(in8, imgInName);

    output_img = gen_output_stack(N*2, points);
    sim_image = uint8(output_img/max(output_img(:))*255);
    imstackwrite(sim_image, imgOutName);
end