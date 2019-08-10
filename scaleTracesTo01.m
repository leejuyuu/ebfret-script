function [tracesOut, scale] = scaleTracesTo01(traces)
% Make fake traces for ebfret fitting
% Pull individual traces in to range by linear scaling max--> 0.9, min
% -->0.1
% Traces (nMolecules x nFrames) double
[nMolecules,~] = size(traces);
tracesOut = zeros(size(traces));
scale = zeros(nMolecules,2);
for iMolecule = 1:nMolecules
    Imax = max(traces(iMolecule,:));
    % put the minimum to either 0 or the negative value
    Imin = min(traces(iMolecule,:));
    if Imin > 0
        Imin = 0;
    end
    % y = mx + b
    mi = (Imax-Imin)/(0.9-0.1);
    bi = Imin - 0.1*mi;
    tracesOut(iMolecule, :) = (traces(iMolecule,:)-bi)/mi;
    scale(iMolecule, :) = [mi, bi];
end


end