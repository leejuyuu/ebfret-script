function Aout = fix_A(A,mus,mus0)
% If the rank order of mus and mus0 is not the same, adjust A so that state
% 1 in Aout is the same as state one in mus0

% get ranks of states
[ig mus0] = sort(mus0);
[ig mus] = sort(mus);

if isequal(mus0,mus)
    Aout = A;
    return
end

K=length(mus);
Aout = zeros(K);

for i=1:K
    for j=1:K
        Aout(mus(i),mus(j)) = A(i,j);
    end
end
