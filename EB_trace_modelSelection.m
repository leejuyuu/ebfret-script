function EB_trace_modelSelection()
load('runstemp0810.mat');
maxK = length(runs(:,1));
nMols = length(runs(1).vb);
nFrames = length(runs(1).vit(1).z);
L = zeros(maxK, nMols);
BIC = zeros(maxK, nMols);
nParams = zeros(1,maxK);
for i = 1:maxK
    L(i,:) = [runs(i).vb.L];
    L(L<0) = 0;
    
    
    BIC(i,:) = -2*log(L(i,:)) + i*(i+5)*log(116);
    AIC(i,:) = -2*log(L(i,:)) + nParams(i)*2;
end



end