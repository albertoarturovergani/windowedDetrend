function report=run_benchmark(fixturePath,outputDir)
% RUN_BENCHMARK() generates a native 10x10 noiseless paper-style grid.
% RUN_BENCHMARK(fixturePath,outputDir) runs the Python-generated shared data,
% including noisy replicates, and verifies numerical agreement.
root=fileparts(mfilename('fullpath'));
if nargin<1, fixturePath=''; end
if nargin<2, outputDir=fullfile(root,'results'); end
if ~exist(outputDir,'dir'), mkdir(outputDir); end
hasReference=~isempty(fixturePath);
if hasReference
 S=load(fixturePath); t=S.times(:); data=S.data; truth=S.truth; parameters=S.parameters;
else
 t=(0:2499)'/10000; rises=linspace(.001,.050,10); decays=linspace(.025,.250,10);
 data=zeros(200,1,numel(t)); truth=zeros(200,numel(t)); parameters=zeros(200,5); i=0;
 for family=0:1
  names={'windowed','biexp'};
  for r=rises
   for d=decays
    i=i+1; [gt,a]=tmspath_synthetic(t,r,d,names{family+1});
    data(i,1,:)=reshape(gt+a,1,1,[]); truth(i,:)=gt'; parameters(i,:)=[r d 0 0 family];
   end
  end
 end
end
methods={'window_exp','window_poly3','window_poly5','global_biexp'};
recovered=zeros(size(data,1),4,numel(t)); rows=zeros(size(data,1)*4,9); pos=0; maxError=0;
mask=t>=.010;
for i=1:size(data,1)
 r=parameters(i,1); d=parameters(i,2); family=parameters(i,5);
 peak=.015; if family==1, peak=.010+r*log1p(d/r); end
 for m=1:4
  cfg=tmspath_config();cfg.fill_mode='none';cfg.onset_s=.010;cfg.peak_s=peak;cfg.peak_search_end_s=.100;
  cfg.rise_model='singlerise';cfg.decay_model='singledecay';cfg.constrain_rise=true;
  if m==2 || m==3
   cfg.rise_model='poly';cfg.decay_model='poly';cfg.constrain_rise=false;
   cfg.decay_degree=3; if m==3, cfg.decay_degree=5; end
  elseif m==4, cfg.mode='global';
  end
  z=tmspath_detrend(data(i,:,:),t,cfg); z=z(:); gt=truth(i,:)'; y=reshape(data(i,1,:),[],1);
  recovered(i,m,:)=reshape(z,1,1,[]);
  mse=mean((z(mask)-gt(mask)).^2); before=mean((y(mask)-gt(mask)).^2); power=mean(gt(mask).^2);
  pos=pos+1; rows(pos,:)=[i-1 family r*1000 d*1000 m mse before mse/power 100*(1-mse/before)];
  if hasReference
   ref=reshape(S.python_recovered(i,m,:),[],1); maxError=max(maxError,max(abs(z-ref)));
  end
 end
 if mod(i,25)==0, fprintf('%d/%d MATLAB cases\n',i,size(data,1)); end
end
T=array2table(rows,'VariableNames',{'case_id','family_id','tau_rise_ms','tau_decay_ms','method_id','mse_recovery','mse_before','nmse','improvement_pct'});
T.method=string(methods(T.method_id))';
writetable(T,fullfile(outputDir,'matlab_metrics.csv'));
report=struct('cases',size(data,1),'rows',height(T),'compared_to_python',hasReference,'max_absolute_difference',maxError,'tolerance',1e-6,'passed',~hasReference || maxError<1e-6);
save(fullfile(outputDir,'matlab_results.mat'),'recovered','t','parameters','report','-v7');
fid=fopen(fullfile(outputDir,'parity_report.json'),'w'); fprintf(fid,'%s',jsonencode(report)); fclose(fid);
disp(report);
assert(report.passed,'Python/MATLAB discrepancy exceeds tolerance');
end
