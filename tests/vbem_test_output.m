ref = '110801-vbem-output-jeb.mat';
new = '110801-vbem-output.mat';

rout = load(ref);
nout = load(new);

N = length(nout.FRET);
dL = arrayfun(@(n) nout.L{n}(end) - rout.L{n}(end), 1:N);
dmu = [nout.w(:).mu] - [rout.w(:).mu];
dsigma = [nout.w(:).W].*[nout.w(:).nu] - [rout.w(:).W].*[rout.w(:).nu];

fprintf('sum(dL): %.2e\n', sum(dL))
fprintf('sum(dmu): %.2e\n', sum(dmu,2))
fprintf('sum(dsigma): %.2e\n', sum(dsigma,2))