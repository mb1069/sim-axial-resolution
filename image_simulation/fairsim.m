% SIM psf calculation for hexSIM with light sheet
% Simulates the raw images produced while scanning an object (3d point 
% cloud) through focus as the hexSIM illumination pattern is shifted 
% through 7 positions laterally.
% Finally it processes the illumination using the hxSimProcessor class to 
% generate superresolved output.
% 
% if not(exist('N', 'var')) && not(exist('npoints', 'var')) && not(exist('showfigures', 'var'))
%     N=256;          % Points to use in FFT
%     npoints=1000;   % Number of random points
%     showfigures = true;
% end
function imgout = fairsim(input_file, output_file)
    showfigures = false;
    N=256;

    pixelsize = 6.5;    % Camera pixel size
    magnification = 60; % Objective magnification
    dx=pixelsize/magnification;     % Sampling in lateral plane at the sample in um
    NA=1.1;         % Numerical aperture at sample
    n=1.33;         % Refractive index at sample
    lambda=0.525;   % Wavelength in um
    rng(1234);      % set random number generator seed
    % eta is the factor by which the illumination grid frequency exceeds the
    % incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
    % resolution without zeros in TF
    % carrier is 2*kmax*eta
    eta=1.0;
    axial=false;     % Whether to model axial or in-plane polarised illumination

    dxn = lambda/(4*NA);          % 2*Nyquist frequency in x and y.
    Nn = ceil(N*dx/dxn/2)*2;      % Number of points at Nyquist sampling, even number
    dxn = N*dx/Nn;                % correct spacing
    res = lambda/(2*NA);
    oversampling = res/dxn;       % factor by which pupil plane oversamples the coherent psf data

    dk=oversampling/(Nn/2);       % Pupil plane sampling
    [kx,ky] = meshgrid(-dk*Nn/2:dk:dk*Nn/2-dk,-dk*Nn/2:dk:dk*Nn/2-dk);

    kr=sqrt(kx.^2+ky.^2);

    % Raw pupil function, pupil defined over circle of radius 1.
    csum=sum(sum((kr<1))); % normalise by csum so peak intensity is 1

    zrange=7;          % distance either side of focus to calculate
    alpha=asin(NA/n);
    dzn=0.8*lambda/(2*n*(1-cos(alpha)));    % Nyquist sampling in z, reduce by 10% to account for gaussian light sheet
    dz=0.4;             % step size in axial direction of PSF
    Nz=2*ceil(zrange/dz);
    dz=2*zrange/Nz;
    Nzn=2*ceil(zrange/dzn);
    dzn=2*zrange/Nzn;
    if Nz < Nzn
        Nz = Nzn;
        dz = dzn;
    end
    clear psf;
    psf=zeros(Nn,Nn,Nzn);
    c=zeros(Nn);
    fwhmz=3;            % FWHM of light sheet in z
    sigmaz=fwhmz/2.355;

    pupil = (kr<1);


    disp('Running SIM');
    img = double(imstackread(input_file));
    %% Reconstruct

    h=hexSimProcessor();
    h.NA=NA;
    h.magnification=magnification;
    h.w=0.05;
    h.beta=0.99;
    h.cleanup=false;
    h.eta=0.8*eta;
    h.axial=axial;
    h.N=N;
    h.debug=false;

    disp("Calibration");
    tic
    % Different frames needed to calibrate low light conditions for
    % struct_10_8 and struct_10_16
%     h.calibrate(img(:,:,17:17+7)+img(:,:,50:57)+img(:,:,100:107));
    h.calibrate(img(:,:,7*Nz/2+1:7*Nz/2+7)+img(:,:,7*Nz/2-13:7*Nz/2-7)+img(:,:,7*Nz/2+15:7*Nz/2+21));
    h.reset();
    toc

    % imsr=single(zeros(400,400,Nz*7));
    imsr=zeros(2*N,2*N,Nz*7,'single');

    disp("Incremental reconstruction");
    tic
    for i = 1:Nz*7
        imo = h.reconstructframe(img(:,:,i),mod(i-1,7)+1);
    %     imsr(:,:,i) = imo(313:712,313:712);
        imsr(:,:,i) = imo(:,:);
    end
    toc

    if showfigures
        figure(8);
        imshow(squeeze(sum(imsr,2)),[]);

        implay(imsr/max(imsr(:)));
    end

    toc
    disp("Calculating z-packed image stack in batch mode");
    tic
    imgout = h.batchreconstruct(img);
    toc
    imstackwrite(imgout, output_file);
end




