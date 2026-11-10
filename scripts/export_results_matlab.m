function export_results_matlab(root)
% Render actual MATLAB benchmark outputs, not Python-recovered waveforms.
if nargin<1, root=fileparts(mfilename('fullpath')); end
S=load(fullfile(root,'results','shared_fixture.mat'));
M=load(fullfile(root,'results','matlab','matlab_results.mat'));
out=fullfile(root,'images');if ~exist(out,'dir'),mkdir(out);end
f=figure('Visible','off','Color','w','Position',[0 0 1300 900]);
tl=tiledlayout(f,2,2,'TileSpacing','compact');
names={'Windowed exponential','Windowed polynomial 3/3','Windowed polynomial 3/5','Global bi-exponential'};
families={'Windowed capacitive artifact','Bi-exponential artifact'};
reds=interp1(linspace(0,1,9),[255 255 255;254 224 210;252 187 161;252 146 114;251 106 74;239 59 44;203 24 29;165 15 21;103 0 13]/255,linspace(0,1,256));
t=M.t(:)*1000;mask=t>=10;ids=[101 208];starts=[1 108];
for col=1:2
 ax=nexttile(tl,col);hold(ax,'on');plot(ax,t,S.truth(ids(col),:),'k','LineWidth',2,'DisplayName','Original oscillator');
 for m=1:4,plot(ax,t,reshape(M.recovered(ids(col),m,:),[],1),'DisplayName',names{m});end
 title(ax,[families{col} ' - 3/200 ms']);xlabel(ax,'Time (ms)');ylabel(ax,'Amplitude (a.u.)');xlim(ax,[0 250]);legend(ax,'Location','best','FontSize',8);
 ax=nexttile(tl,col+2);idx=starts(col)+(0:99);v=squeeze(M.recovered(idx,2,:));
 mse=mean((v(:,mask)-S.truth(idx,mask)).^2,2);
 imagesc(ax,linspace(1,50,10),linspace(25,250,10),reshape(mse,10,10));axis(ax,'xy');set(ax,'ColorScale','log');clim(ax,[.1 1000]);colormap(ax,reds);
 cb=colorbar(ax);cb.Label.String='MSE (a.u.^2), logarithmic';xlabel(ax,'Tau rise (ms)');ylabel(ax,'Tau decay (ms)');title(ax,'Recovery MSE - windowed polynomial 3/3');
end
title(tl,'MATLAB plugin results - bounded solver; known peak; no noise');
exportgraphics(f,fullfile(out,'plugin_results_matlab.png'),'Resolution',180);
exportgraphics(f,fullfile(out,'plugin_results_matlab.pdf'),'ContentType','vector');close(f);
disp('Exported native MATLAB plugin results (PNG and PDF).');
end
