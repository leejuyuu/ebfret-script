function EB_fitting()
traceFileName = 'D:\matlab_CoSMoS\data\20190705\L2_02_01_traces.dat'
traceFile = load(traceFileName, '-mat');
redTraces = traceFile.traces.red;
[redTracesOut, scale] = scaleTracesTo01(redTraces);

redInput = num2cell(redTracesOut',1);
runs = eb_fret(redInput, [1:3], 2);

end
