function report=run_artifact_sweep(fixturePath,outputDir)
% Panel B isolated-artifact sweep; bounded plugin solver, no neural signal.
% Python artifact_sweep.py also runs the original SciPy notebook solver.
root=fileparts(mfilename('fullpath'));
if nargin<1, fixturePath=fullfile(fileparts(root),'results','artifact_only_fixture.mat'); end
if nargin<2, outputDir=fullfile(fileparts(root),'results','matlab'); end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
S=load(fixturePath);t=S.times(:);mask=t>.010;
cfg=tmspath_config();cfg.mode='global';cfg.onset_s=.010;
rows=zeros(size(S.data,1),5);maxError=0;
for i=1:size(S.data,1)
 [~,trend]=tmspath_detrend(S.data(i,:,:),t,cfg); fit=trend(:);
 artifact=reshape(S.data(i,1,:),[],1); ref=S.python_trend(i,:)';
 mse=mean((artifact(mask)-fit(mask)).^2);
 rows(i,:)=[i-1 S.parameters(i,3) S.parameters(i,1)*1000 S.parameters(i,2)*1000 mse];
 maxError=max(maxError,max(abs(fit-ref)));
end
T=array2table(rows,'VariableNames',{'case_id','family_id','tau_rise_ms','tau_decay_ms','mse_artifact_fit'});
writetable(T,fullfile(outputDir,'artifact_only_matlab_metrics.csv'));
report=struct('artifacts',size(S.data,1),'solver','bounded','max_absolute_difference',maxError,'tolerance',1e-6,'passed',maxError<1e-6);
fid=fopen(fullfile(outputDir,'artifact_only_parity.json'),'w');fprintf(fid,'%s',jsonencode(report));fclose(fid);
disp(report);assert(report.passed,'Artifact-only Python/MATLAB parity failed');
end
