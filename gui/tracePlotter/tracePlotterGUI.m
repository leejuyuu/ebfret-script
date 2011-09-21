function varargout = tracePlotterGUI(varargin)
% tracePlotterGUI Visualise a dataset of single-molecule FRET traces 
% along results ofwith VBEM inference.
%
% N.B.: Call using wrapper function tracePlotter.m
%
% Syntax (Strandard GUIDE)
% ------------------------
%
%   H = TRACEPLOTTERGUI returns the handle to a new TRACEPLOTTERGUI.
%
%   TRACEPLOTTERGUI('CALLBACK',hObject,eventData,handles,...) calls the 
%   local function named CALLBACK in TRACEPLOTTERGUI.M with the given 
%   input arguments.
%
%   TRACEPLOTTERGUI('Property','Value',...) creates a new TRACEPLOTTERGUI 
%   Starting from the left, property value pairs are applied to the 
%   GUI before trace_plotter_OpeningFcn gets called. An unrecognized 
%   property name or invalid value makes property application stop.  
%   All inputs are passed to trace_plotter_OpeningFcn via varargin.
%
% Jan-Willem van de Meent
% $Revision: 1.00 $  $Date: 2011/05/16$

% Last Modified by GUIDE v2.5 19-Sep-2011 23:36:06

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @tracePlotterGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @tracePlotterGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before tracePlotter is made visible.
function tracePlotterGUI_OpeningFcn(hObject, eventdata, handles, varargin)
    % Choose default command line output for tracePlotter
    handles.output = hObject;

    % Init plot data and state if not previously called
    % TODO: not sure about behaviour when gui_Singleton = 0
    if ~isfield(handles, 'data')
        handles.data = varargin{end};
    end 
    
    if ~isfield(handles, 'subPlots')
        data = handles.data;
        handles.subPlots = 1;
        handles.rawPlot = 0;
        handles.gamPlot = 0;
        handles.K = 0;
        if isfield(data,'don') & isfield(data,'acc')
            handles.subPlots = handles.subPlots + 1;
            handles.rawPlot = handles.subPlots;
        end
        if isfield(data,'z')
            handles.K = max(cat(1,data.z));
        end
        if isfield(data,'g') & isfield(data,'xi')
            handles.K = size(data(1).g, 2);
            handles.subPlots = handles.subPlots + 1;
            handles.gamPlot = handles.subPlots;
        end
        if handles.subPlots > 1
            P = handles.subPlots;
            for p = 1:P
                handles.plotAxes(p) = subplot('position', [0.04, 0.9*(P-p)/P+0.1, 0.94, 1/P-0.1]);
            end
        end
    end

    if ~isfield(handles, 'state')
        handles.curPlot = 1;
        handles.maxPlot = length(handles.data);
    end

    % Update handles structure
    guidata(hObject, handles);

    % Set Slider Max Position and Stepd
    set(handles.plotSlider, 'Max', handles.maxPlot);
    set(handles.plotSlider, 'SliderStep', [1./handles.maxPlot, 1./handles.maxPlot]);

    % Update plot
    tracePlotterGUI_PlotChanged(hObject, eventdata, handles);

    % UIWAIT makes tracePlotter wait for user response (see UIRESUME)
    % uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = tracePlotterGUI_OutputFcn(hObject, eventdata, handles) 
    % Get default command line output from handles structure
    varargout{1} = handles.output;


% --- Executes every time the current plot is changed
function tracePlotterGUI_PlotChanged(hObject, eventdata, handles)
    % update slider position
    set(handles.plotSlider, 'Value', handles.curPlot);

    % update edit field
    set(handles.plotEdit, 'String', ...
        sprintf('%d / %d', handles.curPlot, ...
                           handles.maxPlot));

    % get data
    d = handles.data(handles.curPlot);

    % plot FRET signal
    axes(handles.plotAxes(1));
    cla();
    T = length(d.FRET);
    t = 1:T;
    plot(t, d.FRET, '-ok');
    xlim([0, T]);
    ylim([-0.2, 1.2+eps]);

    % plot viterbi signal (if specified)
    if isfield(d, 'x') & isfield(d, 'z')
        K = handles.K;
        hold on;

        % loop over states
        c = 0.66 .* hsv(K);
        for k = 1:K
            msk = (d.z(:) == k);
            nmsk = ones(size(msk));
            nmsk(~msk) = NaN;

            % plot x_hat
            l = scatter(t(:) .* nmsk(:), d.x(:) .* nmsk(:), 18, c(k,:), 'Filled');
            set(l, 'LineWidth', 1.5)

            % % plot mu posterior confidence intervals
            % s_mu = sqrt(1 ./ (w.beta(k) .* w.W(k) .* (w.nu(k) - 2)));
            % l = plot(t(:) .* nmsk(:), (x_hat(:) + 2.*s_mu) .* nmsk(:), ... 
            %          '-', 'Color', c(k,:));
            % l = plot(t(:) .* nmsk(:), (x_hat(:) - 2.*s_mu) .* nmsk(:), ...
            %          '-', 'Color', c(k,:));

            % % plot emission confidence intervals
            % s = 1./sqrt(w.nu(k) * w.W(k));
            % l = plot(t(:) .* nmsk(:), (x_hat(:)+2.*s) .* nmsk(:), ...
            %          '-.', 'Color', c(k,:));
            % l = plot(t(:) .* nmsk(:), (x_hat(:)-2.*s) .* nmsk(:), ...
            %          '-.', 'Color', c(k,:));

            % % append to labels
            % mu_labels{k} = sprintf('mu_%d = %.2f +/- %.2f', k, w.mu(k), s_mu);
            % sigma_labels{k} = sprintf('sigma_%d = %.2f +/- %.2f', k, s, 1./sqrt(2.*w.nu(k).*w.W(k).^2));
            % Nk_labels{k} = sprintf('N_%d = %05.1f', k, sum(stat.gamma(:,k)));

            % hold on;
        end

        hold off;
    end


    % plot raw signal
    if handles.rawPlot
        if length(d.don) > T
            T = length(d.don);
            t = 1:T;
            axes(handles.plotAxes(1));
            xlim([0, T]);
        end
        axes(handles.plotAxes(handles.rawPlot));
        cla();
        plot(t, d.don, 'g');
        hold on;
        plot(t, d.acc, 'r');
        xlim([0, T]);
        hold off;
    end

    % if length(fieldnames(d)) > 1
    %     hold on;

    %     % % loop over states
    %     % c = 0.66 .* hsv(K);
    %     % for k = 1:K
    %     %     msk = (z_hat == k)
    %     %     nmsk = ones(size(msk));
    %     %     nmsk(~msk) = NaN;

    %     %     % plot x_hat
    %     %     l = plot(t(:) .* nmsk(:), x_hat(:) .* nmsk(:), '--', 'Color', c(k,:));
    %     %     set(l, 'LineWidth', 1.5)

    %     %     % plot mu posterior confidence intervals
    %     %     s_mu = sqrt(1 ./ (w.beta(k) .* w.W(k) .* (w.nu(k) - 2)));
    %     %     l = plot(t(:) .* nmsk(:), (x_hat(:) + 2.*s_mu) .* nmsk(:), ... 
    %     %              '-', 'Color', c(k,:));
    %     %     l = plot(t(:) .* nmsk(:), (x_hat(:) - 2.*s_mu) .* nmsk(:), ...
    %     %              '-', 'Color', c(k,:));

    %     %     % plot emission confidence intervals
    %     %     s = 1./sqrt(w.nu(k) * w.W(k));
    %     %     l = plot(t(:) .* nmsk(:), (x_hat(:)+2.*s) .* nmsk(:), ...
    %     %              '-.', 'Color', c(k,:));
    %     %     l = plot(t(:) .* nmsk(:), (x_hat(:)-2.*s) .* nmsk(:), ...
    %     %              '-.', 'Color', c(k,:));

    %     %     % append to labels
    %     %     mu_labels{k} = sprintf('mu_%d = %.2f +/- %.2f', k, w.mu(k), s_mu);
    %     %     sigma_labels{k} = sprintf('sigma_%d = %.2f +/- %.2f', k, s, 1./sqrt(2.*w.nu(k).*w.W(k).^2));
    %     %     Nk_labels{k} = sprintf('N_%d = %05.1f', k, sum(stat.gamma(:,k)));

    %     %     hold on;
    %     % end
    % end



    % % add text labels
    % t = text(0.05*T, 1.10, sprintf('L/T = %.1f', L(end)/T), 'FontSize', 14);
    % t = text(0.60*T, 1, Nk_labels, 'FontSize', 14);
    % t = text(0.70*T, 1, mu_labels, 'FontSize', 14);
    % t = text(0.85*T, 1, sigma_labels, 'FontSize', 14);


% --- Executes on slider movement.
function plotSlider_Callback(hObject, eventdata, handles)
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    newPlot = round(get(hObject, 'Value'));
    if newPlot ~= handles.curPlot
        handles.curPlot = newPlot;
        guidata(hObject, handles);
        tracePlotterGUI_PlotChanged(hObject, eventdata, handles);    
    end

% --- Executes during object creation, after setting all properties.
function plotSlider_CreateFcn(hObject, eventdata, handles)
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
      set(hObject,'BackgroundColor',[.9 .9 .9]);
    end



function plotEdit_Callback(hObject, eventdata, handles)
    newPlot = round(str2num(get(hObject,'String')));
    if ~isempty(newPlot)
        if (newPlot >= 1) & (newPlot <= handles.maxPlot)
            if newPlot ~= handles.curPlot
                handles.curPlot = newPlot;
                guidata(hObject, handles);
                tracePlotterGUI_PlotChanged(hObject, eventdata, handles);    
            end
        end
    end

% --- Executes during object creation, after setting all properties.
function plotEdit_CreateFcn(hObject, eventdata, handles)
    % Hint: edit controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
      set(hObject,'BackgroundColor','white');
    end
    % scatter(t, FRET, 8, [0,0,0]);


% --- Executes on button press in prevButton.
function prevButton_Callback(hObject, eventdata, handles)
% hObject    handle to prevButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.curPlot > 1
    handles.curPlot = handles.curPlot - 1;
    guidata(hObject, handles);
    tracePlotterGUI_PlotChanged(hObject, eventdata, handles);    
end

% --- Executes on button press in nextButton.
function nextButton_Callback(hObject, eventdata, handles)
% hObject    handle to nextButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.curPlot < handles.maxPlot
    handles.curPlot = handles.curPlot + 1;
    guidata(hObject, handles);
    tracePlotterGUI_PlotChanged(hObject, eventdata, handles);    
end
