function selection = tmspath_select_offset_channels(data,times,channels,cfg)
% Native equivalent of computeSlopes_v4 and check_detrend_need.
% Population z-score over all epochs/channels in each window, then epoch mean.
% Electrode distances in the source are reporting-only, not selection inputs.
channels=channels(:)'; t=times(:); ne=size(data,1); nc=numel(channels);
selection=struct('policy',cfg.channel_policy,'channels',channels, ...
 'threshold',cfg.slope_threshold,'window_labels',{{'pre','rise','decay'}});
switch cfg.channel_policy
 case 'all'
  offsets=channels;
 case 'manual'
  offsets=channels(ismember(channels,cfg.offset_channels));
 otherwise
  slopes=zeros(ne,nc,3); peaks=zeros(ne,nc);
  for j=1:nc
   for e=1:ne
    y=reshape(data(e,channels(j),:),[],1);
    [pre,rise,decay,peaks(e,j)]=tmspath_time_masks(t,y,cfg);
    masks={pre,rise,decay};
    for k=1:3
     x=t(masks{k}); z=y(masks{k});
     assert(numel(x)>=2,'Automatic channel screening needs two samples in every window');
     x=x-mean(x); slopes(e,j,k)=sum(x.*(z-mean(z)))/sum(x.^2);
    end
   end
  end
  zscores=zeros(size(slopes));
  for k=1:3
   values=reshape(slopes(:,:,k),[],1); sigma=std(values,1);
   if sigma>0, zscores(:,:,k)=(slopes(:,:,k)-mean(values))/sigma; end
  end
  mean_z=reshape(mean(zscores,1),nc,3);
  offsets=channels(any(abs(mean_z)>=cfg.slope_threshold,2));
  selection.slopes=slopes; selection.zscores=zscores;
  selection.mean_z=mean_z; selection.peaks_s=peaks;
end
selection.offset_channels=offsets;
selection.found_offsets=~isempty(offsets);
end
