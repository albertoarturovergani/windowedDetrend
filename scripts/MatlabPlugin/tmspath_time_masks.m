function [pre,rise,decay,peak] = tmspath_time_masks(times,signal,cfg)
% Published TEP boundaries: onset in pre/rise, peak in rise, decay after peak.
% A supplied off-grid peak resolves to the last sample at or before that time.
t=times(:); signal=signal(:); finish=cfg.end_s;
if isempty(finish), finish=t(end); end
peak=cfg.peak_s;
if isempty(peak)
 ids=find(t>=cfg.onset_s & t<=min(cfg.peak_search_end_s,finish));
 assert(~isempty(ids),'Empty peak search interval');
 [~,k]=max(abs(signal(ids))); peak=t(ids(k));
end
assert(cfg.onset_s<=peak && peak<finish,'Peak must lie after onset and before correction end');
if strcmp(cfg.mask_convention,'pipeline')
 ids=find(t>=cfg.onset_s & t<=peak);
 assert(~isempty(ids),'Rise window contains no samples'); peak=t(ids(end));
 pre=t<=cfg.onset_s; rise=t>=cfg.onset_s & t<=peak; decay=t>peak & t<=finish;
else
 pre=t<cfg.onset_s; rise=t>=cfg.onset_s & t<peak; decay=t>=peak & t<=finish;
end
end
