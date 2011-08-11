function [A0, C] = normalize(A, dim)
% [A0, C] = normalize(A, dim)
%
% Returns normalized array A0 and normalization const C. Array is
% along single axis if dim is specified, or along all axes if not.
if nargin < 2
  C = sum(A(:));
  A = A ./ (C + (C==0));  
else
  C = sum(A, dim);
  A = bsxfun(@rdivide, A, C + (C==0));
end