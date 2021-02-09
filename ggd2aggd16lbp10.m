function [feat28] = ggd2aggd16lbp10(img)

    %% ggd2
    [alpha, sigma] = estimateggdparam(img(:));
    feat_ggd2 = [alpha, sigma^2]; 
    
    %% aggd16
    shifts = [0 1; 1 0; 1 1; -1 1];
    feat_aggd16 = [];
    for itr_shift = 1:4 %along four orientations
        shifted = circshift(img, shifts(itr_shift, :));
        pair = img(:).*shifted(:);       
        % [N{i}, edges{i}] = hist(pair(:), 500);
        [alpha leftstd rightstd] = estimateaggdparam(pair); %estimate AGGD parameters
        const                    = (sqrt(gamma(1/alpha))/sqrt(gamma(3/alpha)));
        meanparam                = (rightstd-leftstd)*(gamma(2/alpha)/gamma(1/alpha))*const;
        feat_aggd16                 = [feat_aggd16 alpha meanparam leftstd^2 rightstd^2];        
    end

    %% lbp10
    P = 8;
    R = 1;
    lbp_type = { 'ri', 'u2', 'riu2' };
    y = 3;
    mtype = lbp_type{y};
    mapping = getmapping(P, mtype); 
    feat_lbp10 = lbp_new(img, R, P, mapping, 'nh');
    
    %% all features 
    feat28 = [feat_ggd2, feat_aggd16, feat_lbp10];
    
end

