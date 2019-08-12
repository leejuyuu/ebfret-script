function [newVb, newVit, selection] = selectK(runs)
load('runstemp0810.mat');
maxK = length(runs(:,1));
nMols = length(runs(1).vb);
selection = ones(1,nMols);
for iMol = 1:nMols
    for iK = maxK:-1:1
        
        % Criterion 1: the fitted states should be separated by at least 3
        % sigma 
        sigma = 1./(sqrt(runs(iK).vb(iMol).w.W.*runs(iK).vb(iMol).w.nu));
        mu = runs(iK).vb(iMol).w.mu;
        if any(mu(1:end-1)+2.5*sigma(1:end-1) >= mu(2:end)) ||...
                any(mu(2:end)-2.5*sigma(2:end) <= mu(1:end-1))                
            continue
        else
            % Criterion 2: any fitted state should occupy at least 40
            % frames in total.
            isSmallOccupancy = false;
            for i = 1:iK
                isSmallOccupancy = sum(runs(iK).vit(iMol).z == i) < 40;
            if isSmallOccupancy
                break
            end
            end
            if isSmallOccupancy
                continue
            end
            
            selection(iMol) = iK;
            break
        end
    end
end
newVb = runs(1).vb;
newVit = runs(1).vit;

for i = 1:maxK
    equalK = selection == i;
    newVb(equalK) = runs(i).vb(equalK);
    newVit(equalK) = runs(i).vit(equalK);
    
end
   
end
