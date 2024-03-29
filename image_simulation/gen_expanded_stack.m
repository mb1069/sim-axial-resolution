% SIM psf calculation for hexSIM with light sheet
% Simulates the raw images produced while scanning an object (3d point 
% cloud) through focus as the hexSIM illumination pattern is shifted 
% through 7 positions laterally.
% Finally it processes the illumination using the hxSimProcessor class to 
% generate superresolved output.

% if not(exist('N', 'var')) && not(exist('npoints', 'var')) && not(exist('showfigures', 'var'))
%     N=256;          % Points to use in FFT
%     npoints=1000;   % Number of random points
%     showfigures = true;
% end
% 
%% Load simulated chromatin structures

data_shape = size(points);
npoints = data_shape(1);

points(:,3) = points(:,3) - mean(points(:,3));
max_z = max(points(:,3));
min_z = abs(min(points(:,3)));

% zrange = ceil(max(max_z, min_z)) + 1; % distance either side of focus to calculate
zrange = 7;

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

%% Calculate 3d PSF

nz = 1;
disp("Calculating 3d psf");

tic
for z = -zrange:dzn:zrange-dzn
    c(pupil) = exp(1i*(z*n*2*pi/lambda*sqrt((1-kr(pupil).^2*NA^2/n^2))));
    psf(:,:,nz) = abs(fftshift(ifft2(c))).^2*exp(-z^2/2/sigmaz^2);
    nz = nz+1; 
end

% Normalised so power in resampled psf (see later on) is unity in focal plane
psf = psf * Nn^2/sum(pupil(:))*Nz/Nzn; 

toc

if showfigures
    figure(1);
    imshow(psf(:,:,Nzn/2+10),[]);
end

%% Calculate 3D-OTF

disp("Calculating 3d otf");

tic
otf = fftn(psf);
toc

aotf = abs(fftshift(otf));
m = max(aotf(:));
if showfigures
    figure(2);
    imshow(log(squeeze(aotf(Nn/2+1,:,:))+0.0001),[]);

    figure(3);
    imshow(log(squeeze(aotf(:,Nn/2+1,:))+0.0001),[]);

    figure(4);
    plot(squeeze(sum(aotf,[1 2])));
end

if showfigures
    figure(20);
    pcshow(points);
end
%% Generate phase tilts in frequency space

xyrange = Nn/2*dxn;
dkxy = pi/xyrange;
kxy = -Nn/2*dkxy:dkxy:(Nn/2-1)*dkxy;
dkz = pi/zrange;
kz = -Nzn/2*dkz:dkz:(Nzn/2-1)*dkz;

phasetilts=complex(single(zeros(Nn,Nn,Nzn,7)));

disp("Calculating pointwise phase tilts");

tic
% parfor j = 1:7
for j = 1:7
    pxyz = complex(single(zeros(Nn,Nn,Nzn)));
    for i = 1:npoints
        x=points(i,1);
        y=points(i,2);
        z=points(i,3)+dz/7*(j-1); 
        ph=eta*4*pi*NA/lambda;
        p1=-j*2*pi/7;
        p2=j*4*pi/7;
        if axial % axial polarisation normalised to peak intensity of 1
            ill = 2/9*(3/2+cos(ph*(y)+p1-p2)...
                +cos(ph*(y-sqrt(3)*x)/2+p1)...
                +cos(ph*(-y-sqrt(3)*x)/2+p2));
        else     % in plane polarisation normalised to peak intensity of 1
            ill = 2/9*(3-cos(ph*(y)+p1-p2)...
                -cos(ph*(y-sqrt(3)*x)/2+p1)...
                -cos(ph*(-y-sqrt(3)*x)/2+p2));
        end
        px = exp(1i*single(x*kxy));
        py = exp(1i*single(y*kxy));
        pz = exp(1i*single(z*kz))*ill;
        pxy = px.'*py;
        for ii = 1:length(kz)
            pxyz(:,:,ii) = pxy.*pz(ii);
        end
        phasetilts(:,:,:,j) = phasetilts(:,:,:,j)+pxyz;
    end
end
toc

%% calculate output

disp("Calculating raw image stack");

tic

img = zeros(N,N,Nz*7,'single');

for j = 1:7
    ootf = fftshift(otf) .* phasetilts(:,:,:,j);
    img(:,:,j:7:end) = abs(ifftn(ootf,[N N Nz])); 
            % OK to use abs here as signal should be all positive.
            % Abs is required as the result will be complex as the 
            % fourier plane cannot be shifted back to zero when oversampling.
            % But should reduction in sampling be allowed here (Nz<Nzn)?
end
toc

if showfigures
    figure(5);
    imshow(sum(img,3),[]);

    figure(6);
    imshow(squeeze(sum(img,2)),[]);

    implay(img/max(img(:)));
end


if exist('nphot', 'var')
    if nphot > -1
        disp("Adding noise")
        disp(nphot)
        % Add poisson noise and recalculate, uncomment to simulate noisy data
%         nphot = 100; % expected number of photons at brightest points in image

        tic
        img = poissrnd(img*nphot);
        toc
    end
end
% imsr=single(zeros(400,400,Nz*7));




