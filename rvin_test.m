clear; clc;
addpath('utilities');
addpath('./matconvnet-1.0-beta24/matlab/');
run ('./matconvnet-1.0-beta24/matlab/vl_setupnn.m');

folderTest='./Set12/';
noiseSigma  = 30;  %%% image noise level
use_gpu = 1;


for ii = 135000:5000:135000
    model_deploy=strcat('NIDCN_mat_RVIN.prototxt');
    model_weights=strcat('./RVIN_30/RVIN_solver__iter_',num2str(ii),'.caffemodel');
    ext         =  {'*.tif','*.png','*.bmp'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    end
    for i = 1:length(filePaths)
        label = imread(fullfile(folderTest,filePaths(i).name));
        [H,W,Z]=size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        disp([num2str(i),'    ',filePaths(i).name,'    ',num2str(noiseSigma)]);
        if(size(label,3)>1)
            label = rgb2ycbcr(label);
            label=label(:, :, 1);
        end
        
        label = im2double(label);

        M= label;
        xx = rand(size(label));
        p3 = 0.3;
        N=xx;
        N(N>=p3)=0;
        N1 = N;
        N1 = N1(N1>0);
        Imn=min(N1(:));
        Imx=max(N1(:));
        N=((N-Imn)./(Imx-Imn));%.*(255-0))
        M(xx<p3) = N(xx<p3);
        im_input = M;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        if use_gpu
            caffe.reset_all();
            caffe.set_mode_gpu();
            caffe.set_device(0);
        else
            caffe.reset_all();
            caffe.set_mode_cpu();
        end
        net = caffe.Net(model_deploy, model_weights, 'test');
        scores = net.forward({im_input});
        output = scores{1};
        figure(i+333);imshow(output,[]);

    end
end