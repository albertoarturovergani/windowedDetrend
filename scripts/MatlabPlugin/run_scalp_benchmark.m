function report = run_scalp_benchmark(fixtureDir, outputDir, caseIds)
% Compare all 64 channels of every Python-generated case using native MATLAB.
% Fixture: data[1,64,2500], reference[methods,64,2500], shared metadata/truth.
% Optional caseIds selects zero-based case IDs; use separate output directories
% for disjoint batches. A subset report describes only the selected cases.
root=fileparts(mfilename('fullpath'));
if nargin<1, fixtureDir=fullfile(root,'..','results','scalp64'); end
if nargin<2, outputDir=fullfile(fixtureDir,'matlab'); end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
reportPath=fullfile(outputDir,'parity_report.json');
if exist(reportPath,'file'), delete(reportPath); end
summary=jsondecode(fileread(fullfile(fixtureDir,'summary.json')));
assert(summary.channels==64 && summary.samples==2500,'Unexpected dataset dimensions');
manifest=jsondecode(fileread(fullfile(fixtureDir,'manifest.json')));
assert(numel(manifest)==summary.cases,'Manifest case count mismatch');
if nargin>=3
 assert(~isempty(caseIds) && numel(unique(caseIds))==numel(caseIds),'Invalid subset');
 assert(all(ismember(caseIds,[manifest.case_id])),'Unknown case IDs');
 manifest=manifest(ismember([manifest.case_id],caseIds));
end
M=load(fullfile(fixtureDir,'metadata.mat'));
configs=jsondecode(fileread(fullfile(fixtureDir,'method_configs.json')));
t=M.times(:); mask=t>=.010; methods=M.method_names;
assert(isequal(size(M.truth),[64 numel(t)]) && numel(methods)==5,'Invalid metadata');
rows=zeros(numel(manifest)*64*numel(methods),10); pos=0; maximum=0;
started=tic;
for i=1:numel(manifest)
 S=load(fullfile(fixtureDir,manifest(i).file));
 assert(S.case_id==manifest(i).case_id,'Case ID mismatch');
 assert(isequal(size(S.data),[1 64 numel(t)]),'Invalid input shape');
 assert(isequal(size(S.python_recovered),[numel(methods) 64 numel(t)]),'Invalid reference shape');
 assert(all(isfinite(S.data(:))) && all(isfinite(S.python_recovered(:))),'Nonfinite fixture');
 recovered=zeros(numel(methods),64,numel(t));
 for m=1:numel(methods)
  cfg=configs.(methods{m}); cfg.peak_s=S.peak_s;
  clean=tmspath_detrend(S.data,t,cfg);
  assert(all(isfinite(clean(:))),'Nonfinite MATLAB output');
  recovered(m,:,:)=clean;
  for c=1:64
   z=reshape(clean(1,c,:),[],1); gt=M.truth(c,:)'; y=reshape(S.data(1,c,:),[],1);
   ref=reshape(S.python_recovered(m,c,:),[],1); delta=z-ref;
   err=max(abs(delta)); maximum=max(maximum,err);
   mse=mean((z(mask)-gt(mask)).^2); before=mean((y(mask)-gt(mask)).^2);
   % SciPy saves case_id as int64. Cast it before concatenation, otherwise
   % MATLAB coerces the entire row (including MSE/differences) to integers.
   pos=pos+1; rows(pos,:)=[double(S.case_id) c-1 m mse before mse/mean(gt(mask).^2) ...
       err mean(delta(mask).^2) mean((z(t>=.010 & t<.030)-gt(t>=.010 & t<.030)).^2) ...
       mean((z(t>=.030)-gt(t>=.030)).^2)];
  end
 end
 % Retain native waveforms for both noiseless extreme examples; all channels
 % and cases are checked above and retain per-channel metrics below.
 if strcmp(manifest(i).group,'extreme') && manifest(i).noise_sd==0
  save(fullfile(outputDir,sprintf('case_%03d_results.mat',S.case_id)),'recovered','t','-v7');
 end
 fprintf('%d/%d cases x 64 channels x 5 methods; max difference %.3g\n',i,numel(manifest),maximum);
end
T=array2table(rows,'VariableNames',{'case_id','channel_id','method_id','mse_recovery', ...
 'mse_before','nmse','max_absolute_difference','output_difference_mse','mse_early','mse_late'});
T.method=string(methods(T.method_id))'; T.channel=string(M.channel_names(T.channel_id+1))';
assert(max(rows(:,7))==maximum,'Metric storage lost numerical precision');
writetable(T,fullfile(outputDir,'matlab_metrics.csv'));
save(fullfile(outputDir,'matlab_metrics.mat'),'rows','methods','-v7');
report=struct('cases',numel(manifest),'dataset_cases',summary.cases, ...
 'case_ids',[manifest.case_id],'channels',64,'samples',numel(t), ...
 'traces',numel(manifest)*64,'reconstructed_traces',height(T), ...
 'max_absolute_difference',maximum,'tolerance',1e-6, ...
 'passed',maximum<1e-6,'seconds',toc(started),'matlab_version',version);
fid=fopen(reportPath,'w'); assert(fid~=-1,'Cannot write report');
fprintf(fid,'%s',jsonencode(report)); fclose(fid); disp(report);
assert(report.passed,'Python/MATLAB discrepancy exceeds tolerance');
end
