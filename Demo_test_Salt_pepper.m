clear; clc;
addpath('utilities');
addpath('./matconvnet-1.0-beta24/matlab/');
run ('./matconvnet-1.0-beta24/matlab/vl_setupnn.m');

folderTest='./Set12/';
folderModel = 'model_30/';
noiseSigma  = 30;  %%% image noise level


for ii = 150000:5000:150000

    load(fullfile(folderModel, ['Salt_pepper_1__iter_' num2str(ii) '.mat']));

    ext         =  {'*.tif','*.png','*.bmp'};
    filePaths   =  [];
    
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    end
  
    
    for i = 1:length(filePaths)
        %%% read images
        label = imread(fullfile(folderTest,filePaths(i).name));
        [H,W,Z]=size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        disp([num2str(i),'    ',filePaths(i).name,'    ',num2str(noiseSigma)]);
        if(size(label,3)>1)
            label = rgb2ycbcr(label);
            label=label(:, :, 1);
        end
        
        label = im2double(label);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        M = label;
        xx = rand(size(label));
        p3 = 0.3;
        M(xx < p3/2) = 0; % Minimum value
        M(xx >= p3/2 & xx < p3) = 1; % Maximum (saturated) value
        input = M;

        output=NIDCN_SP(input,model);

        figure;imshow(output,[]);

    end

end

