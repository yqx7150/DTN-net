run matconvnet/matlab/vl_setupnn ;

vl_compilenn('enableGpu', true, ...
'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0', ...  %change it 
'cudaMethod', 'nvcc',...
'enableCudnn',false,... 
'cudnnroot','local/cuda');
%}
warning('off');