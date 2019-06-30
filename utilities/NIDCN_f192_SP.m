function im_h_y = NIDCN_f192_SP(im_l_y,model)
 
 weight = model.weight;
 bias = model.bias;
 
 
layer_num = size(weight,2);


im_y = single(im_l_y);

%disp(size(im_y)); 

convfea_1 = vl_nnconv(im_y,weight{7},bias{7},'Pad',3);

convfea = vl_nnconv(convfea_1,weight{1},bias{1},'Pad',3);
convfea = vl_nnrelu(convfea);

%disp(size(convfea));

convfea = vl_nnconv(convfea,weight{2},bias{2},'Pad',3);
convfea = vl_nnrelu(convfea);

%disp(size(convfea));
 
        
         
convfea = vl_nnconv(convfea,weight{3},bias{3},'Pad',3);
convfea = vl_nnrelu(convfea);

%disp(size(convfea));

convfea = vl_nnconv(convfea,weight{4},bias{4},'Pad',3);
convfea = vl_nnrelu(convfea);

%disp(size(convfea));
 
convfea = vl_nnconv(convfea,weight{5},bias{5},'Pad',3);
%disp(size(convfea));

 
 
im_h_y_1 = convfea + convfea_1;

% convfea_6 = vl_nnconv(im_h_y_1,weight{6},bias{6},'Pad',3);
% 
% convfea_7 = vl_nnconv(convfea_6,weight{7},bias{7},'Pad',3);
% convfea_7 = vl_nnrelu(convfea_7);
%      
% convfea_8 = vl_nnconv(convfea_7,weight{8},bias{8},'Pad',3);
% convfea_8= vl_nnrelu(convfea_8);
% 
% convfea_9 = vl_nnconv(convfea_8,weight{9},bias{9},'Pad',3);
% 
% im_h_y_2 = convfea_6 + convfea_9;
% 
% convfea_10 = vl_nnconv(im_h_y_2,weight{10},bias{10},'Pad',3);
% 
% im_h_y_3 = convfea_6 + convfea_10;

im_h_y = vl_nnconv(im_h_y_1,weight{6},bias{6},'Pad',3);
 %disp(size(im_h_y)); 
    
end
