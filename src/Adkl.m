function [dkl Anorm] = Adkl(Aobs,A0)

A0=normalise(A0,2)';
Aobs=normalise(Aobs,2)';
q0=null(A0-eye(length(A0)));q0=q0/sum(q0(:));
% 4/8/10
[L W] = size(q0);
if L~=W
    q0 = normalise(diag(A0^10000));
end
dkl=sum(A0.*log((realmin+A0)./(realmin+Aobs)),1)*q0;

eobs =(Aobs-A0)./(eps+A0);
Anorm = norm(eobs);
