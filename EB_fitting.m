function EB_fitting()
tic
maxState = 4;
restarts = 3;

traceFileName = 'D:\matlab_CoSMoS\data\20190705\L2_02_01_traces.dat';
traceFile = load(traceFileName, '-mat');
redTraces = traceFile.traces.red;
[redTracesOut, redscale] = scaleTracesTo01(redTraces);

redInput = num2cell(redTracesOut',1);
redruns = eb_fret(redInput, [1:maxState], restarts);
toc
% save('runstemp0810','runs');
[redVb, redVit, redSelection] = selectK(redruns);
redVit = scaleVitBack(redVit, redscale);
toc

greenTraces = traceFile.traces.green;
[greenTracesOut, greenscale] = scaleTracesTo01(greenTraces);

greenInput = num2cell(greenTracesOut',1);
greenruns = eb_fret(greenInput, [1:maxState], restarts);
toc
% save('runstemp0810','runs');
[greenVb, greenVit, greenSelection] = selectK(greenruns);
greenVit = scaleVitBack(greenVit, greenscale);

save('test0811_ebparam','greenruns','redruns','greenVit','redVit','greenSelection','redSelection');


end
