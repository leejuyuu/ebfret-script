N = length(FRET);
data = cell(1,N);
path = cell(1,N);
labels = cell(1,N);
r=10;
r=size(x_hat,1)

LPsum = zeros(1,K);
LPhist_T = zeros(K,N);
for k=1:K
    for n=1:N
        LPsum(k) = LPsum(k) + out{r,k}{n}.F(end);
        LPhist_T(k,n) = out{r,k}{n}.F(end)/length(FRET{n});
    end
end

[ig k] = max(LPsum)
%keyboard
return 
for n = 1:N
    data{n} = [1-FRET{n}(:) FRET{n}(:)];
    path{n} = x_hat{r,k}{n};
    labels{n} = sprintf('trace %03d: r%02d k%d',n,r,k);
end

save(sprintf('%s_k%d_r%02d_hFRETgui',dname,k,r),'data','path','labels');
%save('foo','data','path');