function im_h_y = BN_NIDCN_SP(im_l_y,model)

weight = model.weight;
bias = model.bias;

bnmean= model.bnmean;
bnvariance=model.bnvariance;
bnscale=model.bnscale ;
scaleG=model.scaleG ;
scaleB=model.scaleB ;
eps=1e-5;

layer_num = size(weight,2);
%disp(layer_num);

im_y = single(im_l_y);

%disp(size(im_y));




convfea_1 = vl_nnconv(im_y,weight{7},bias{7},'Pad',3);

convfea = vl_nnconv(convfea_1,weight{1},bias{1},'Pad',3);


bnme=repmat(reshape(bnmean{1},1,1,size(bnmean{1},1)),size(convfea,1),size(convfea,2));
scaG=repmat(reshape(scaleG{1},1,1,size(scaleG{1},1)),size(convfea,1),size(convfea,2));
std=repmat(reshape(sqrt(bnvariance{1}/bnscale{1} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
scaB=repmat(reshape(scaleB{1},1,1,size(scaleB{1},1)),size(convfea,1),size(convfea,2));
x=convfea-bnme / bnscale{1};
for i = size(bnmean{1},1):-1:1
    y(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
end


convfea = vl_nnrelu(y);

for m=2:4
    convfea = vl_nnconv(convfea,weight{m},bias{m},'Pad',3);
    
    bnme=repmat(reshape(bnmean{m},1,1,size(bnmean{m},1)),size(convfea,1),size(convfea,2));
    scaG=repmat(reshape(scaleG{m},1,1,size(scaleG{m},1)),size(convfea,1),size(convfea,2));
    std=repmat(reshape(sqrt(bnvariance{m}/bnscale{m} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
    scaB=repmat(reshape(scaleB{m},1,1,size(scaleB{m},1)),size(convfea,1),size(convfea,2));
    x=convfea-bnme / bnscale{m};
    for i = size(bnmean{m},1):-1:1
        y(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
    end
    
    convfea = vl_nnrelu(y);
    
    %   disp(size(convfea));
    
end


convfea = vl_nnconv(convfea,weight{5},bias{5},'Pad',3);

bnme=repmat(reshape(bnmean{5},1,1,size(bnmean{5},1)),size(convfea,1),size(convfea,2));
scaG=repmat(reshape(scaleG{5},1,1,size(scaleG{5},1)),size(convfea,1),size(convfea,2));
std=repmat(reshape(sqrt(bnvariance{5}/bnscale{5} + eps),1,1,size(convfea,3)),size(convfea,1),size(convfea,2));
scaB=repmat(reshape(scaleB{5},1,1,size(scaleB{5},1)),size(convfea,1),size(convfea,2));
x=convfea-bnme / bnscale{5};
for i = size(bnmean{5},1):-1:1
    yy(:,:,i) = scaG(:,:,i).* x(:,:,i) ./ std(:,:,i) +scaB(:,:,i);
end


im_h_y_1 = yy + convfea_1;
im_h_y = vl_nnconv(im_h_y_1 ,weight{6},bias{6},'Pad',3);

end
