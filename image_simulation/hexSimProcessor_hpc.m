classdef hexSimProcessor < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        N=512;          % Points to use in FFT
        pixelsize = 6.5;    % Camera pixel size
        magnification = 60; % Objective magnification
        NA=1.1;         % Numerical aperture at sample
        n=1.33;         % Refractive index at sample
        lambda=0.525;   % Wavelength in um
        alpha=0.3;      % Zero order attenuation width
        beta=0.999;      % Zero order attenuation
        % eta is the factor by which the illumination grid frequency exceeds the
        % incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
        % resolution without zeros in TF
        % carrier is 2*kmax*eta
        eta=1.;
        w=0.05;         % Wiener parameter
        cleanup=true;
        debug=false;
        axial=false;     % if axial/radial polarised illumination is used
        usemodulation=true;     % if axial/radial polarised illumination is used
    end
    properties (Access = private) 
        dx;         % Sampling in image plane
        dk;         % Sampling in frequency plane
        k;
        sum_separated_comp;
        reconfactor;    % for reconstruction
        prefilter;     % for prefilter stage, includes otf and zero order supression
        prefilters;     % for prefilter stage, includes otf and zero order supression shifted
        postfilter;   % post filter
        imgbig;    % space for oversampled images
        carray;     % complex array for reconstruction
        imgstore;
        bigimgstore;
    end
    
    methods
        function obj = hexSimProcessor()
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            
        end
        
        function calibrate(o,img)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            o.dx = o.pixelsize/o.magnification;     % Sampling in lateral plane at the sample in um
            res=o.lambda/(2*o.NA);
            oversampling=res/o.dx;    % factor by which pupil plane oversamples the coherent psf data

            o.dk=oversampling/(o.N/2);    % Pupil plane sampling
            o.k=-o.dk*o.N/2:o.dk:o.dk*o.N/2-o.dk;
            [lkx,lky] = meshgrid(o.k,o.k);

            kr=sqrt(lkx.^2+lky.^2);

            % Separate bands into DC and 3 high frequency bands
            
            M=(exp(1i*2*pi/7).^((0:3)'*(0:6)));

            o.sum_separated_comp=zeros(o.N,o.N,7);
            
            for b = 1:4
                for i = 1:7
                    o.sum_separated_comp(:,:,b) = o.sum_separated_comp(:,:,b) + img(:,:,i) * M(b,i);
                end
            end
            
            m = kr>1.9*o.eta; % minimum search radius in k-space 
                        
            % Find parameters
%             tic
            for i = 1:3
                [kx(i),ky(i),p(i),ampl(i)] = o.findCarrier(i,m);
            end
%             toc
            
            % Pre-calculate reconstruction factors
            o.reconfactor=single(zeros(2*o.N,2*o.N,7));

            dx2 = o.dx/2;
%             [x2,y2] = meshgrid(-dx2*o.N:dx2:dx2*o.N-dx2,-dx2*o.N:dx2:dx2*o.N-dx2);

            j=1;
            ph=2*pi*o.NA/o.lambda;
            if o.axial
                A=6;
            else 
                A=12;
            end
            
            xx=-dx2*o.N:dx2:dx2*o.N-dx2;
            yy=xx.';
            for pstep = [0:6]*2*pi/7
                if o.usemodulation
                    o.reconfactor(:,:,j) = (1 + 4/ampl(1)*real(exp(1i*ph*ky(1)*yy)*exp(1i*(ph*kx(1)*xx-pstep+p(1))))...
                        + 4/ampl(2)*real(exp(1i*ph*ky(2)*yy)*exp(1i*(ph*kx(2)*xx-2*pstep+p(2))))...
                        + 4/ampl(3)*real(exp(1i*ph*ky(3)*yy)*exp(1i*(ph*kx(3)*xx-3*pstep+p(3))))); 
                else
                    o.reconfactor(:,:,j) = (1 + A*real(exp(1i*ph*ky(1)*yy)*exp(1i*(ph*kx(1)*xx-pstep+p(1))))...
                        + A*real(exp(1i*ph*ky(2)*yy)*exp(1i*(ph*kx(2)*xx-2*pstep+p(2))))...
                        + A*real(exp(1i*ph*ky(3)*yy)*exp(1i*(ph*kx(3)*xx-3*pstep+p(3))))); 
                end% signs do not match theory yet!
%                 o.reconfactor(:,:,j) = (1 + A*cos(ph*(kx(1)*x2+ky(1)*y2)-pstep+p(1))...
%                         + A*cos(ph*(kx(2)*x2+ky(2)*y2)-2*pstep+p(2))...
%                         + A*cos(ph*(kx(3)*x2+ky(3)*y2)-3*pstep+p(3))); % signs do not match theory yet!
                j=j+1;
            end
            
            % calculate pre-filter factors
            o.prefilter = single(zeros(o.N));
            m = (kr<2);
            o.prefilter(m) = single(o.tf(kr(m)).*o.att(kr(m)));
            o.prefilter = fftshift(o.prefilter);

            % calculate wiener filter
            [kxbig,kybig] = meshgrid(-o.dk*o.N:o.dk:o.dk*o.N-o.dk,-o.dk*o.N:o.dk:o.dk*o.N-o.dk);
            wienerfilter = zeros(2*o.N);
            mtot = false(2*o.N);
            for i = 1:3
                kr=sqrt((kxbig-kx(i)).^2+(kybig-ky(i)).^2);
                m = (kr<2);
                mtot = mtot | m;
                wienerfilter(m) = wienerfilter(m) + o.tf(kr(m)).^2.*o.att(kr(m));
                kr=sqrt((kxbig+kx(i)).^2+(kybig+ky(i)).^2);
                m = (kr<2);
                mtot = mtot | m;
                wienerfilter(m) = wienerfilter(m) + o.tf(kr(m)).^2.*o.att(kr(m));
            end
            kr=sqrt(kxbig.^2+kybig.^2);
            m = (kr<2);
            mtot = mtot| m;
            wienerfilter(m) = wienerfilter(m) + o.tf(kr(m)).^2.*o.att(kr(m));

            if o.debug
                figure(1000);
                imshow(wienerfilter,[]);
            end
            
            kmax = 1*(2+sqrt(kx(1)^2+ky(1)^2));
            wienerfilter(mtot) = (1-kr(mtot)/kmax)./(wienerfilter(mtot)+o.w^2);
            o.postfilter = fftshift(single(wienerfilter));

            o.carray = complex(zeros(2*o.N,'single'),0);
            o.imgbig = single(zeros(2*o.N,2*o.N,7));    % space for oversampled images
            
            if o.cleanup
                imgo = o.reconstruct(img);
                m=imdilate(abs(fftshift(fft2(imgo)))>10*imgaussfilt(abs(fftshift(fft2(imgo))),5),ones(5));
                m(o.N-11:o.N+13,o.N-11:o.N+13)=false(25);
                o.postfilter(fftshift(m))=0;
            end
            o.reset();
        end
        
        function [atf] = att(o, kr)
            atf = (1 - o.beta*exp(-kr.^2/(2*o.alpha^2)));
        end
            
        function [otf] = tf(o, kr)
            otf = (1/pi*(acos(kr/2) - kr/2.*sqrt(1-kr.^2/4)));
        end
            
        function [imgout] = reconstruct(o, img)
            for i = 1:7         % prefilter and oversample
                imf = fft2(img(:,:,i));
                o.carray(1:o.N/2,1:o.N/2)=imf(1:o.N/2,1:o.N/2).*o.prefilter(1:o.N/2,1:o.N/2);
                o.carray(1:o.N/2,3*o.N/2+1:end)=imf(1:o.N/2,o.N/2+1:end).*o.prefilter(1:o.N/2,o.N/2+1:end);
                o.carray(3*o.N/2+1:end,1:o.N/2)=imf(o.N/2+1:end,1:o.N/2).*o.prefilter(o.N/2+1:end,1:o.N/2);
                o.carray(3*o.N/2+1:end,3*o.N/2+1:end)=imf(o.N/2+1:end,o.N/2+1:end).*o.prefilter(o.N/2+1:end,o.N/2+1:end);
                o.imgbig(:,:,i) = ifft2(o.carray,'symmetric');
            end
            
            img2 = sum(o.imgbig.*o.reconfactor,3);    % reconstruct and post filter
            o.imgstore = img;
            o.bigimgstore = ifft2(fft2(img2).*o.postfilter,'symmetric');
            imgout = o.bigimgstore;
        end
        
        function [imgout] = reconstructframe(o, img, i)
            diff = img - o.imgstore(:,:,i);
            imf = fft2(diff).*o.prefilter;
            o.carray(1:o.N/2,1:o.N/2)=imf(1:o.N/2,1:o.N/2);
            o.carray(1:o.N/2,3*o.N/2+1:end)=imf(1:o.N/2,o.N/2+1:end);
            o.carray(3*o.N/2+1:end,1:o.N/2)=imf(o.N/2+1:end,1:o.N/2);
            o.carray(3*o.N/2+1:end,3*o.N/2+1:end)=imf(o.N/2+1:end,o.N/2+1:end);
            diffbig = ifft2(o.carray,'symmetric');
            
            img2 = diffbig.*o.reconfactor(:,:,mod(i-1,7)+1);    % reconstruct and post filter
            o.imgstore(:,:,i) = img;
            o.bigimgstore = o.bigimgstore + ifft2(fft2(img2).*o.postfilter,'symmetric');
            imgout = o.bigimgstore;
        end
        
        function [imgout] = batchreconstruct(o,img)
            nim = size(img,3);
            r = mod(nim,14);
            if r>0          % pad with empty frames so total number of frames is divisible by 14
                img = cat(3,img,zeros(o.N,o.N,14-r,'like',img));
                nim = nim+14-r;
            end
            nim7 = nim/7;
            imf = fft2(img).*o.prefilter(:,:,ones(1,nim));
            bcarray = zeros(o.N*2,o.N*2,7,'like',complex(single(0)));
            img2 = zeros(o.N*2,o.N*2,nim,'single');
            for i = 1:7:nim
                bcarray(1:o.N/2,1:o.N/2,:)=imf(1:o.N/2,1:o.N/2,i:i+6);
                bcarray(1:o.N/2,3*o.N/2+1:end,:)=imf(1:o.N/2,o.N/2+1:end,i:i+6);
                bcarray(3*o.N/2+1:end,1:o.N/2,:)=imf(o.N/2+1:end,1:o.N/2,i:i+6);
                bcarray(3*o.N/2+1:end,3*o.N/2+1:end,:)=imf(o.N/2+1:end,o.N/2+1:end,i:i+6);
                img2(:,:,i:i+6) = ifft2(bcarray,'symmetric').*o.reconfactor;
            end
            if o.debug
                fs = abs(fftshift(fftn(img2)));
                figure(1012);
                imshow(log(squeeze(abs(fs(self.N+1,:,:)))+10),[]);
            end
            img2fs = fft(img2,nim,3);
            clear img2;
            block1 = img2fs(:,:,[1:nim7/2]);
            s = size(block1);
            middle_block = zeros([s(1), s(2), 80]);
            block2 = img2fs(:,:,[end-(nim7/2-1):end]);
            padded_img2fs = cat(3, block1, middle_block, block2);
            
            imgout = ifft(padded_img2fs,120,3,'symmetric');
            clear imf2fs;
            imgout = ifft2(fft2(imgout).*o.postfilter,'symmetric');
        end
        
        function [imgout] = batchreconstructcompact(o,img)
            nim = size(img,3);
            r = mod(nim,14);
            if r>0          % pad with empty frames so total number of frames is divisible by 14
                img = cat(3,img,zeros(o.N,o.N,14-r,'like',img));
                nim = nim+14-r;
            end
            nim7 = nim/7;
            imf = fft2(img).*o.prefilter(:,:,ones(1,nim));
            bcarray = zeros(o.N*2,o.N*2,7,'like',complex(single(0)));
            img2 = zeros(o.N*2,o.N*2,nim,'single');
            for i = 1:7:nim
                bcarray(1:o.N/2,1:o.N/2,:)=imf(1:o.N/2,1:o.N/2,i:i+6);
                bcarray(1:o.N/2,3*o.N/2+1:end,:)=imf(1:o.N/2,o.N/2+1:end,i:i+6);
                bcarray(3*o.N/2+1:end,1:o.N/2,:)=imf(o.N/2+1:end,1:o.N/2,i:i+6);
                bcarray(3*o.N/2+1:end,3*o.N/2+1:end,:)=imf(o.N/2+1:end,o.N/2+1:end,i:i+6);
                img2(:,:,i:i+6) = ifft2(bcarray,'symmetric').*o.reconfactor;
            end
            if o.debug
                fs = abs(fftshift(fftn(img2)));
                figure(1012);
                imshow(log(squeeze(abs(fs(self.N+1,:,:)))+10),[]);
            end
            imgout = zeros(o.N*2,o.N*2,nim7,'single');
            imf = fft(img2(1:o.N,1:o.N,:),nim,3);
            imgout(1:o.N,1:o.N,:) = ifft(imf(:,:,[1:nim7/2, end-(nim7/2-1):end]),nim7,3,'symmetric');
            imf = fft(img2(o.N+1:o.N*2,1:o.N,:),nim,3);
            imgout(o.N+1:o.N*2,1:o.N,:) = ifft(imf(:,:,[1:nim7/2, end-(nim7/2-1):end]),nim7,3,'symmetric');
            imf = fft(img2(1:o.N,o.N+1:o.N*2,:),nim,3);
            imgout(1:o.N,o.N+1:o.N*2,:) = ifft(imf(:,:,[1:nim7/2, end-(nim7/2-1):end]),nim7,3,'symmetric');
            imf = fft(img2(o.N+1:o.N*2,o.N+1:o.N*2,:),nim,3);
            imgout(o.N+1:o.N*2,o.N+1:o.N*2,:) = ifft(imf(:,:,[1:nim7/2, end-(nim7/2-1):end]),nim7,3,'symmetric');
            imgout = ifft2(fft2(imgout).*o.postfilter,'symmetric');
            whos
        end
        
        function reset(o)
            o.imgstore = zeros(o.N,o.N,21,'single');
            o.bigimgstore = zeros(2*o.N,'single');
        end
        
        function [kx, ky, phase, ampl] = findCarrier(o, band, mask)
            ix=o.sum_separated_comp(:,:,band+1).*o.sum_separated_comp(:,:,1);
            ixf=abs(fftshift(fft2(fftshift(ix))));
            [pxc,pyc] = o.findPeak((ixf-imgaussfilt(ixf,20)).*mask);
            if o.debug
                figure(1010+band)
                imshow(sqrt(ixf),[]);
                drawcircle('Center',[pxc,pyc],'Radius',3,'Color','red','InteractionsAllowed','none');
            end
            [ixfz,Kx,Ky] = o.zoomf(ix,o.N,o.k(pxc),o.k(pyc),50,o.dk*o.N);
            if o.debug
                figure(1020+band)
                imshow(abs(ixfz),[]);
            end
            [pxcf,pycf] = o.findPeak(abs(ixfz));
%             phase = angle(ixfz(pycf,pxcf))
            kx = Kx(pxcf);
            ky = Ky(pycf);
            
            % For getting the modulation depth:

            band0_img = o.sum_separated_comp(:,:,1);
            band1_img = o.sum_separated_comp(:,:,band+1);

            otf_exclude_min_radius = 0.5; % Min Radius of the circular region around DC that is to be excluded from the cross-correlation calculation
            otf_exclude_max_radius = 1.5; % Max Radius of the circular region around DC that is to be excluded from the cross-correlation calculation

            [lkx,lky] = meshgrid(o.k,o.k);
            kr=sqrt(lkx.^2+lky.^2);

            m = (kr<2);
            otf = ones(o.N,o.N,'single')*0.01;
            otf(m) = single(o.tf(kr(m)));
            otf = fftshift(otf);

            otf_mask = (kr>otf_exclude_min_radius) & (kr<otf_exclude_max_radius);
            otf_mask_for_band_common_freq = fftshift(otf_mask & imtranslate(otf_mask,[pxc-(o.N/2+1),pyc-(o.N/2+1)]));
                
            if o.debug
                figure(2000+band);
                imshow(fftshift(otf_mask_for_band_common_freq));
                figure(2020+band);
                imshow(otf);
            end
                               
            band0_common = ifft2(fft2(o.sum_separated_comp(:,:,1))./otf.*otf_mask_for_band_common_freq);
            
            phase_shift_to_xpeak = exp(-1i*kx*(-o.N/2*o.dx:o.dx:o.N/2*o.dx-o.dx)*2*pi*o.NA/o.lambda); % may need to change the sign of kx
            phase_shift_to_ypeak = exp(-1i*ky*(-o.N/2*o.dx:o.dx:o.N/2*o.dx-o.dx)*2*pi*o.NA/o.lambda).';  % may need to change the sign of ky

            band1_common = ifft2(fft2(o.sum_separated_comp(:,:,band+1))./otf.*otf_mask_for_band_common_freq).*(phase_shift_to_ypeak * phase_shift_to_xpeak);
              
            scaling = 1/sum(band0_common(:).*conj(band0_common(:)));
                 
            cross_corr_result = sum(band0_common .* band1_common,'all') * scaling;
            
            if o.debug
                figure(2040+band);
                imshow(sqrt(abs(fftshift(fft2(band0_common .* band1_common)))),[]);
                figure(2050+band);
                imshow(abs(fftshift(fft2(o.sum_separated_comp(:,:,1))./otf.*otf_mask_for_band_common_freq)),[]);
                figure(2060+band);
                imshow(abs(fftshift(fft2(o.sum_separated_comp(:,:,band+1))./otf.*otf_mask_for_band_common_freq)),[]);
            end
                
            ampl = abs(cross_corr_result) * 2;
            phase = angle(cross_corr_result);
                
        end
        
        function [px,py] = findPeak(o, arr)
            %finds the position of the largest value in a 2-d array
            [~,I] = max(arr(:));
            dim = size(arr,1);
            py = mod(I-1,dim)+1;
            px = floor((I-1)/dim)+1;
        end

        function [res, kxarr, kyarr] = zoomf(o,arr,M,kx,ky,mag,kmax)
            %produces a zoomed in view of k space
            resy = czt(arr,M,exp(-1i*2*pi/(mag*M)),exp(-1i*pi*(1/mag-2*ky/kmax)));
            res = czt(resy.',M,exp(-1i*2*pi/(mag*M)),exp(-1i*pi*(1/mag-2*kx/kmax))).';
            kyarr = -kmax*(1/mag-2*ky/kmax)/2 + (kmax/(mag*M))*(0:M-1);
            kxarr = -kmax*(1/mag-2*kx/kmax)/2 + (kmax/(mag*M))*(0:M-1);
            n = size(arr,1);
            % remove phase tilt from (0,0) offset in spatial domain 
            res = res .* exp(1i*(kxarr)*n*pi/kmax);
            res = res .* exp(1i*(kyarr')*n*pi/kmax);        
        end
    end
end
