clc, clear, close all

%% Read a stereoscopic 3D image
imL = imread('test1L.bmp');
imR = imread('test1R.bmp');

%% Binocular semantic feature extraction
addpath(genpath('matconvnet-1.0-beta25/matlab'))
run matconvnet-1.0-beta25/matlab/vl_setupnn;

net = load('matconvnet-1.0-beta25/imagenet-vgg-f.mat');
net = vl_simplenn_tidy(net);

imL_ = single(imL); % note: 255 range
imL_ = imresize(imL_, net.meta.normalization.imageSize(1:2));
imL_ = imL_ - net.meta.normalization.averageImage;

resL = vl_simplenn(net, imL_);

feat_tempL = resL(18).x;

for j = 1:4096
    feat_tempL1(j) = feat_tempL(:, :, j);
end

semantic_featL = feat_tempL1;

imR_ = single(imR); % note: 255 range
imR_ = imresize(imR_, net.meta.normalization.imageSize(1:2));
imR_ = imR_ - net.meta.normalization.averageImage;

resR = vl_simplenn(net, imR_);

feat_tempR = resR(18).x;


for j = 1:4096
    feat_tempR1(j) = feat_tempR(:, :, j);
end

semantic_featR = feat_tempR1;

binocular_semantic_feat = 0.5*double(semantic_featL) + 0.5*double(semantic_featR);


%% Binocular quality-aware feature extraction
L1 = double(rgb2gray(imL));
R1 = double(rgb2gray(imR));
         
h = fspecial('gaussian', 7, 1);

L2 = conv2(L1, h, 'valid');
R2 = conv2(R1, h, 'valid');

L2 = L2(1:2:end, 1:2:end);
R2 = R2(1:2:end, 1:2:end);

L3 = conv2(L2, h, 'valid');
R3 = conv2(R2, h, 'valid');

L3 = L3(1:2:end, 1:2:end);
R3 = R3(1:2:end, 1:2:end);

L1_n = divisiveNormalization(L1);
L2_n = divisiveNormalization(L2);
L3_n = divisiveNormalization(L3);

R1_n = divisiveNormalization(R1);
R2_n = divisiveNormalization(R2);
R3_n = divisiveNormalization(R3); 

quality_featL = [ggd2aggd16lbp10(L1_n), ggd2aggd16lbp10(L2_n), ggd2aggd16lbp10(L3_n)];
quality_featR = [ggd2aggd16lbp10(R1_n), ggd2aggd16lbp10(R2_n), ggd2aggd16lbp10(R3_n)];

binocular_quality_feat = 0.5*quality_featL + 0.5*quality_featR;


%% Compute quality score
load semantic_svr_model.mat
load quality_svr_model.mat
load mu_sigma.mat

n1 = size(semantic_mu, 2);
n2 = size(quality_mu, 2);

for i = 1:n1
    if semantic_sigma(i) ~= 0
        binocular_semantic_feat(:, i) = (binocular_semantic_feat(:, i) - semantic_mu(i))./semantic_sigma(i);
    else 
        binocular_semantic_feat(:, i) = 0;
    end
end

for i = 1:n2
    if quality_sigma(i) ~= 0
        binocular_quality_feat(:, i) = (binocular_quality_feat(:, i) - quality_mu(i))./quality_sigma(i);
    else 
        binocular_quality_feat(:, i) = 0;
    end
end

[binocular_semantic_score, accuracy1, decision_values1] = svmpredict(1, binocular_semantic_feat, semantic_svr_model);
[binocular_quality_score, accuracy2, decision_values2] = svmpredict(1, binocular_quality_feat, quality_svr_model);

predict_score = binocular_semantic_score*0.4 + binocular_quality_score*0.6

% predicted score on test0 stereopair: -7.8086
% predicted score on test1 stereopair: -1.0394
% predicted score on test2 stereopair: 24.8024
% predicted score on test3 stereopair: 41.1825
