%% Multichannel EEG: threshold first
% Run all sections in order. Synthetic EEG, not experimental validation.
% Base MATLAB only; no plugin installation or extra toolboxes needed.
pluginDir=fileparts(mfilename('fullpath')); addpath(pluginDir);

%% A. Controls in order of importance
cfg=tmspath_config();
% 1. Slope Threshold: main decision parameter for extreme-artifact detection.
% A channel score >= threshold is classified as extreme: enable windowed fits.
% Threshold above ALL channel scores: apply the basic whole-epoch linear fit.
% Scores are absolute epoch-mean standardized slopes. Lower => more flagged.
cfg.slope_threshold=0.5;
% 2. Peak search interval, seconds relative to the recorded TMS trigger.
% The detected peak splits rise/decay; slopes use pre, rise AND decay.
cfg.onset_s=0.0;
cfg.peak_search_end_s=0.020;
% 3. Models, kept at the published Windowed TEP defaults.
cfg.pre_degree=2;
cfg.rise_model='lagrange'; % Interpolates every rise sample; degree N-1.
cfg.decay_model='poly';
cfg.decay_degree=3;       % Offset channels: cubic; others: degree 2.
cfg.fill_mode='gaussian'; % Default: replace zeroed rise plus margin with Gaussian noise.
% All other fields keep their library defaults: channel_policy='auto',
% constrain_decay=false, only_offset_channels=false, overall_degree=1.
% No flagged channel => whole-epoch linear fit, no Lagrange and no filling.
% At least one flagged => windowed fits on ALL selected channels by default.
% Changing cfg does not change the synthetic data below.

%% B. Synthetic EEG: channels, stimuli and jitter
toy=struct('n_epochs',12,'n_channels',16,'fs_hz',2000,'seed',42, ...
 'stimulus_jitter_s',0.0007, ... % Common latency per stimulus: +/-0.7 ms.
 'channel_jitter_s',0.00035, ... % Additional latency per channel: +/-0.35 ms.
 'amplitude_jitter',0.10, ...   % Artifact amplitude varies +/-10% per trace.
 'noise_sd',0.05);             % Independent noise, arbitrary units.
[data,times,truth,artifact,metadata]=tmspath_demo_data(toy);
labels=metadata.channel_labels;
% Two channels have stronger artifacts, with weaker contamination elsewhere.
% A damped response and ongoing background provide known clean ground truth.
% Jitter changes signal latency relative to fixed event markers at t=0;
% it does not resample epochs or introduce the removed pulse artifact.
fprintf('Input shape (epochs, channels, samples): %d x %d x %d\n',size(data));
disp('Stimulus jitter (ms):'); disp(1000*metadata.stimulus_jitter_s');

%% 1. Apply automatic screening and Polynomial detrending
[corrected,trend,diagnostics]=tmspath_detrend(data,times,cfg);
selection=diagnostics.channel_selection;
scores=max(abs(selection.mean_z),[],2);
offset_channels=selection.offset_channels;
operation=diagnostics.operation;
mask=times>=0;
error_array=corrected(:,:,mask)-truth(:,:,mask);
mse_full=mean(error_array(:).^2);
fprintf('Operation: %s\n',operation);
disp('Offset channels:'); disp(labels(offset_channels));
disp('Channel  max|mean z-slope|  classification');
for c=1:toy.n_channels
 classification='below threshold'; if ismember(c,offset_channels), classification='offset'; end
 fprintf('%-7s  %8.4f          %s\n',labels{c},scores(c),classification);
end
fprintf('Post-trigger recovery MSE: %.6f a.u.^2\n',mse_full);
% Inspect operation before accessing rise/decay: fallback records have overall.
if strcmp(operation,'windowed')
 degrees=[];
 for k=1:numel(diagnostics.fits)
  rec=diagnostics.fits{k}; if rec.epoch==1, degrees(end+1)=rec.decay.degree; end %#ok<SAGROW>
 end
 disp('Decay degree by channel:'); disp(degrees);
else
 disp('No windowed detrending: whole-epoch linear fit; no gap filling.');
end

%% 2. Change only the threshold on the SAME data
% This sweep illustrates decisions; it does not optimize the threshold.
sweep_thresholds=unique([.25 .5 1 2 3 cfg.slope_threshold]);
sweep_counts=zeros(size(sweep_thresholds));
sweep_operations=cell(size(sweep_thresholds)); sweep_mse=zeros(size(sweep_thresholds));
disp('Threshold  flagged channels  operation             recovery MSE');
for k=1:numel(sweep_thresholds)
 sweep_cfg=cfg; sweep_cfg.slope_threshold=sweep_thresholds(k);
 [sweep_output,~,sweep_info]=tmspath_detrend(data,times,sweep_cfg);
 sweep_counts(k)=numel(sweep_info.channel_selection.offset_channels);
 sweep_operations{k}=sweep_info.operation;
 error_array=sweep_output(:,:,mask)-truth(:,:,mask);
 sweep_mse(k)=mean(error_array(:).^2);
 fprintf('%8.3f  %16d  %-20s  %.6f\n',sweep_thresholds(k),sweep_counts(k),sweep_operations{k},sweep_mse(k));
end

%% 3. Inspect screening and one recovered channel
fig=figure('Color','w','Position',[100 100 1200 800]);
layout=tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
blue=[.14 .35 .47]; orange=[.71 .32 .16]; grey=[.46 .54 .60];
ax=nexttile(layout); bars=bar(ax,1:toy.n_channels,scores,'FaceColor','flat');
bars.CData=repmat(grey,toy.n_channels,1);
bars.CData(offset_channels,:)=repmat(orange,numel(offset_channels),1);
hold(ax,'on'); yline(ax,cfg.slope_threshold,'--',sprintf('Threshold = %g',cfg.slope_threshold),'Color',orange);
xticks(ax,1:toy.n_channels); xticklabels(ax,labels); xtickangle(ax,60); ax.FontSize=9;
ylabel(ax,'max |mean slope z-score|'); title(ax,'1. Channel decision');
ax=nexttile(layout);
plot(ax,sweep_thresholds,sweep_counts,'o','Color',blue,'MarkerFaceColor',blue); hold(ax,'on');
for k=1:numel(sweep_thresholds)
 text(ax,sweep_thresholds(k),sweep_counts(k)+.8,num2str(sweep_counts(k)), ...
  'HorizontalAlignment','center','FontSize',9);
end
xline(ax,cfg.slope_threshold,'--','Color',orange);
ylim(ax,[-1 toy.n_channels+3]); xlim(ax,[min(sweep_thresholds)-.15 max(sweep_thresholds)+.2]);
xlabel(ax,'Slope threshold'); ylabel(ax,'Flagged channels'); title(ax,'2. Same EEG, different threshold');
text(ax,.98,.95,{'Flagged channels > 0: Windowed','No flagged channels: Linear'}, ...
 'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top','FontSize',9);
display_epoch=1; % One epoch, not an average; Python index 0 is the same epoch.
[~,display_channel]=max(scores);
ax=nexttile(layout,[1 2]); c=display_channel; e=display_epoch;
y=reshape(truth(e,c,:),[],1); z=reshape(corrected(e,c,:),[],1);
plot(ax,times*1000,y,'k','DisplayName','Known clean signal'); hold(ax,'on');
plot(ax,times*1000,z,'Color',orange,'DisplayName','Polynomial');
limits=ylim(ax); search_ms=1000*[cfg.onset_s cfg.peak_search_end_s];
patch(ax,search_ms([1 2 2 1]),limits([1 1 2 2]),[.86 .91 .94], ...
 'FaceAlpha',.4,'EdgeColor','none','DisplayName','Peak search');
subtitle='whole-epoch linear';
for k=1:numel(diagnostics.fits)
 rec=diagnostics.fits{k};
 if rec.epoch==e && rec.channel==c && isfield(rec,'peak_s')
  xline(ax,1000*rec.peak_s,':','Color',blue,'DisplayName','Detected split');
  subtitle=sprintf('decay degree %d',rec.decay.degree); break;
 end
end
title(ax,sprintf('%s, epoch %d: %s',labels{c},e,subtitle));
xlabel(ax,'Time from trigger (ms)'); ylabel(ax,'Amplitude (a.u.)'); xlim(ax,[-20 200]);
wave_legend=legend(ax,'Orientation','horizontal','NumColumns',2,'FontSize',9,'Box','off');
wave_legend.Layout.Tile='south';
title(layout,sprintf('Polynomial | %d channels x %d stimuli | %s',toy.n_channels,toy.n_epochs,operation));

%% 4. Butterfly view: before and after correction
% Both panels show ALL channels from the SAME epoch, with the same colors.
% Highlight the two strongest simulated artifacts, independently of screening.
% This display-only use of synthetic metadata does not affect fitting.
[~,ranking]=sort(mean(abs(metadata.artifact_gains),1),'descend');
highlight_channels=ranking(1:2);
other_channels=setdiff(1:toy.n_channels,highlight_channels);
highlight_colors=[orange;blue];
display_mask=times>=-.020 & times<=.200;
fig_butterfly=figure('Color','w','Position',[100 100 1200 480]);
butterfly_layout=tiledlayout(fig_butterfly,1,2,'TileSpacing','compact','Padding','compact');
headings={'Before correction: observed EEG','After correction: Polynomial'};
arrays={data,corrected};
for panel=1:2
 ax=nexttile(butterfly_layout); hold(ax,'on'); values=arrays{panel};
 for j=1:numel(other_channels)
  c=other_channels(j); trace=plot(ax,times*1000,reshape(values(display_epoch,c,:),[],1), ...
   'Color',[.65 .69 .71],'LineWidth',.8);
  if j==1, trace.DisplayName=sprintf('Other channels (%d)',numel(other_channels));
  else, trace.HandleVisibility='off'; end
 end
 for j=1:numel(highlight_channels)
  c=highlight_channels(j);
  plot(ax,times*1000,reshape(values(display_epoch,c,:),[],1),'Color',highlight_colors(j,:), ...
   'LineWidth',1.2,'DisplayName',labels{c});
 end
 xline(ax,0,':','Color',grey,'LineWidth',.8,'HandleVisibility','off');
 visible=values(display_epoch,:,display_mask); lower=min(visible(:)); upper=max(visible(:));
 pad=.08*max(upper-lower,1e-12);
 title(ax,headings{panel}); xlabel(ax,'Time from trigger (ms)'); ylabel(ax,'Amplitude (a.u.)');
 xlim(ax,[-20 200]); ylim(ax,[lower-pad upper+pad]); grid(ax,'on'); ax.GridAlpha=.15;
end
butterfly_legend=legend(ax,'Orientation','horizontal','NumColumns',3,'FontSize',9,'Box','off');
butterfly_legend.Layout.Tile='south';
title(butterfly_layout,{sprintf('Butterfly EEG | %d channels, epoch %d',toy.n_channels,display_epoch), ...
 'Amplitude axes scaled separately; original signal units'});

%% 5. Continue the host pipeline
% Replace data/times with real preprocessed epochs; keep matching good EEG
% channels in MATLAB and Python because screening depends on that population.
% Gaussian filling remains enabled; inspect diagnostics, then continue ICA.
data_for_ica=corrected;
