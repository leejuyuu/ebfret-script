function [out cheatPrior]= cheat_out(mun,covn,zPath,Aobs);

N = length(mun);
K = length(mun{1});
out = cell(1,N);
for n=1:N
    counts = zeros(1,K);
    for k=1:K
        counts(k) = sum(zPath{n} == k);
    end
    out{n}.Wa = Aobs{n}{1};
    out{n}.Wpi = zeros(1,K);
    out{n}.Wpi(zPath{n}(1)) = 1;
    out{n}.beta = counts';
    out{n}.m = mun{n};
    out{n}.v = counts';
    out{n}.W = 1./(covn{n}.*out{n}.v');
    out{n}.F = 1;
end

cheatPrior.upi = zeros(1,3);
cheatPrior.mu = zeros(1,3);
cheatPrior.beta = zeros(1,3)';
cheatPrior.v = zeros(1,3)';
cheatPrior.W = zeros(1,3);
cheatPrior.ua = zeros(3);
