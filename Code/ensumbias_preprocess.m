% 
% % %   Like Benucci et al, 2013. Two ensembles: first is uniform; second has
% % %   2x as many presentations of the 180 deg stimulus.
% 
% %%
clear all
data_folder=fullfile('E:\Documents\MATLAB\NEV and EXPO files');

% load('129r001p173') %OVERWROTE, need to re-read from nev
% load('130l001p169')

% spikes=nev_reader('130l001p169-s.nev',0);
% a=ReadExpoXML('130l001#169[ensunbias1].xml');
% spikes=nev_reader('129r001p173-s.nev',0);
% a=ReadExpoXML('129r001#173[ensunbias1].xml');
% spikes=nev_reader('140l001p107-s.nev',0);
% a=ReadExpoXML('140l001#107[ensunbias1].xml');
% spikes=nev_reader('140l001p122-s.nev',0);
% a=ReadExpoXML('140l001#122[ensunbias1].xml');
% spikes=nev_reader('140r001p105-s.nev',0);
% a=ReadExpoXML('140r001#105[ensunbias1].xml');
% spikes=nev_reader('140r001p122-s.nev',0);
% a=ReadExpoXML('140r001#122[ensunbias1].xml');
% % % % % % % % % % % % 141 files
% spikes=nev_reader(fullfile(data_folder,'141r001p114-s.nev'),0);
% a=ReadExpoXML('141r001#114[ensunbias3].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p006-s.nev'),0);
% a=ReadExpoXML('141r001#006[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p007-s.nev'),0);
% a=ReadExpoXML('141r001#007[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p009-s.nev'),0);
% a=ReadExpoXML('141r001#009[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p024-s.nev'),0);
% a=ReadExpoXML('141r001#024[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p025-s.nev'),0);
% a=ReadExpoXML('141r001#025[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p027-s.nev'),0);
% a=ReadExpoXML('141r001#027[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p038-s.nev'),0);
% a=ReadExpoXML('141r001#038[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p039-s.nev'),0);
% a=ReadExpoXML('141r001#039[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p041-s.nev'),0);
% a=ReadExpoXML('141r001#041[ensunbias3_awaketime].xml');
% % % % % % % % % % % % % % 142 files
% spikes=nev_reader(fullfile('142l001p002-s.nev'),0);
% a=ReadExpoXML('142l001#002[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile('142l001p007-s.nev'),0);
% a=ReadExpoXML('142l001#007[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile('142l001p006-s.nev'),0);
% a=ReadExpoXML('142l001#006[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile('142l001p010-s.nev'),0);
% a=ReadExpoXML('142l001#010[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'142l001p004-s.nev'),0);
% a=ReadExpoXML('142l001#004[ensunbias3_awaketime_fineori].xml');
spikes=nev_reader(fullfile(data_folder,'142l001p009-s.nev'),0);
a=ReadExpoXML('142l001#009[ensunbias3_awaketime_fineori].xml');

stim=a.passes.BlockIDs;
uniq_stim=unique(stim);
syncs=spikes(spikes(:,1)==2000,3);
% syncs=spikes(spikes(:,1)==2000,:); % for when syncs don't match with stim
% syncs=syncs(syncs(:,2)==-3,3);
% syncs=[syncs(1)-diff(syncs(2:3));syncs]; % if missing a sync, verify than do this
%% chop stuff up

blks=find(diff(syncs)>0.1);

index=1;
for i=1:96
    resp=spikes(spikes(:,1)==i,:);
    sort_codes=unique(resp(:,2));
    sort_codes=sort_codes(sort_codes~=0 & sort_codes~=255);
%     if isempty(sort_codes)~=1
        for j=1:length(sort_codes)
            resp2=resp(resp(:,2)==sort_codes(j),3);
            resp2=resp2-0.045;   %shift; ideally should choose this using Matt's method
            for k=1:length(syncs)
                resp_count(index,k)=length(find(resp2>syncs(k) & resp2<syncs(k)+diff(syncs(3:4))));
            end
            index=index+1;
        end
%     end
    disp(i)
end
% 
% index=1;
% for i=2:length(stim)    %because first stim is blank and has no sync
%     if stim(i)~=2
%         temp=a.passes.events{i}.Data{7};
%         ori_stim(index)=temp(2);
%         temp2=a.passes.events{i}.Data{4};
%         con=temp2(end);
%         if con==0
%             ori_stim(index)=200;
%         end
%         index=index+1;
%     else
%         if isnan(ori_stim(end))~=1    %only count the blanks once since there is no sync for these
%             ori_stim(index)=NaN;
%             index=index+1;
%         end
%     end
% end
% ori_stim=ori_stim(1:end-1);     %drop last blank because it has no sync either
% ori_stim=ori_stim(isnan(ori_stim)~=1);
% 
% base=[];bias=[];
% resp_raw_base=[];resp_raw_bias=[];
% for i=1:12
%     if mod(i,2)~=0
%         %         base=[base ori_stim((i-1)*2500+1200:i*2500)];              %looks at only last half of the block
%         %         resp_raw_base=[resp_raw_base resp_count(:,(i-1)*2500+1200:i*2500)];
%         base=[base ori_stim((i-1)*2500+1:i*2500)];
%         resp_raw_base=[resp_raw_base resp_count(:,(i-1)*2500+1:i*2500)];
%     else
%         %         bias=[bias ori_stim((i-1)*2500+1200:i*2500)];
%         %         resp_raw_bias=[resp_raw_bias resp_count(:,(i-1)*2500+1200:i*2500)];
%         bias=[bias ori_stim((i-1)*2500+1:i*2500)];
%         resp_raw_bias=[resp_raw_bias resp_count(:,(i-1)*2500+1:i*2500)];
%     end
% end

akstop
% save as *_preprocess
%% load preprocessed

clear
% load('129r001p173_preprocess')
% load('130l001p169_preprocess')
% load('140l001p107_preprocess')
% load('140l001p122_preprocess')
% load('140r001p105_preprocess')
load('140r001p122_preprocess')

%% calculate stuff

% base tuning
ori_base=[(0:20:160) 200];
% mean response of each unit to each ori and blank stim
for i=1:size(resp_raw_base,1)
    for j=1:length(ori_base)
        temp=find(base==ori_base(j));
        resp_base(i,j)=mean(resp_raw_base(i,temp));
        temp=find(bias==ori_base(j));
        resp_bias(i,j)=mean(resp_raw_bias(i,temp));
    end
end
% resp_bias(:,1)=mean([resp_bias(:,1) resp_bias(:,10)],2);
% resp_bias=[resp_bias(:,1:9) resp_bias(:,11)];

% group neurons by their ori pref
for i=1:size(resp_base,1)
    [~,~,oribias(i),oripref(i),~,~] = orivecfit((0:20:160),resp_base(i,1:9),resp_base(i,10));
end
oripref(oripref>180)=oripref(oripref>180)-360;
oripref(oripref<0)=oripref(oripref<0)+180;  %this makes 0 preferring cells fire most to a 0 stimulus in EXPO

% mean and raw response of each orientation bin (all units with similar
%        ori pref) to each stim
bins=(0:20:180);
for i=1:length(bins)-1
    temp=find(oripref>bins(i) & oripref<bins(i+1));
    ncount(i)=length(temp);
    resp_base_binned(i,:)=sum(resp_base(temp,:),1);             %oripref x 10 test oris
    resp_bias_binned(i,:)=sum(resp_bias(temp,:),1);
    resp_raw_base_binned(i,:)=sum(resp_raw_base(temp,:),1);     %oripref x 15000
    resp_raw_bias_binned(i,:)=sum(resp_raw_bias(temp,:),1);
end

clear dir* temp* ans i index j k quad*

% make benucci et al pre and post predictions (Fig 1c,d,g,h,k,l)
bias(bias==180)=0;
for j=1:length(base)
    pred_resp_basex(:,j)=resp_bias_binned(:,find(ori_base==base(j)));
    pred_resp_base(:,j)=resp_base_binned(:,find(ori_base==base(j)));
    pred_resp_biasx(:,j)=resp_base_binned(:,find(ori_base==bias(j)));
    pred_resp_bias(:,j)=resp_bias_binned(:,find(ori_base==bias(j)));
end

%count number of each stimuli
for i=1:length(ori_base)
    base_count(i)=length(find(base==ori_base(i)));
    bias_count(i)=length(find(bias==ori_base(i)));  %bias redefined to have 180=0 above
end

%normalize
norm_fact=mean(resp_raw_base_binned,2);
for i=1:9
    resp_base_norm(i,:)=resp_raw_base_binned(i,:)./norm_fact(i);
    pred_resp_base_norm(i,:)=mean(pred_resp_base(i,:))/norm_fact(i);
    pred_resp_basex_norm(i,:)=mean(pred_resp_basex(i,:))/norm_fact(i);
    resp_bias_norm(i,:)=resp_raw_bias_binned(i,:)/norm_fact(i);
    pred_resp_bias_norm(i,:)=mean(pred_resp_bias(i,:))/norm_fact(i);
    pred_resp_biasx_norm(i,:)=mean(pred_resp_biasx(i,:))/norm_fact(i);
end

%% do properly: compute linear filter for each ori preference bin

%responses
% resp_raw_base_binned
% resp_raw_bias_binned

% design
% base
% bias

trials=(200:2501);  %throws out the first 200 frames or 10 seconds, which is how long equalization takes in Benucci
ntrials=length(trials);
tdelays=(-1:5);

base_design=zeros(10,ntrials);
bias_design=zeros(10,ntrials);
for i=1:ntrials
    base_design(find(ori_base==base(trials(i))),i)=1;
    bias_design(find(ori_base==bias(trials(i))),i)=1;
end
base_design=base_design';
bias_design=bias_design';

for qq=1:length(tdelays)

    resp_raw_base_binned2=resp_raw_base_binned(:,trials+tdelays(qq))';
    resp_raw_bias_binned2=resp_raw_bias_binned(:,trials+tdelays(qq))';

    b_base(qq,:,:)=pinv(base_design'*base_design)*base_design'*resp_raw_base_binned2;
    b_bias(qq,:,:)=pinv(bias_design'*bias_design)*bias_design'*resp_raw_bias_binned2;

    disp(qq)
end
b_base=b_base/length(tdelays);
b_bias=b_bias/length(tdelays);

% figure out non-linearity
resp_raw_base_binned2=resp_raw_base_binned(:,trials)';
resp_raw_bias_binned2=resp_raw_bias_binned(:,trials)';

for i=2:size(base_design)-5   %stim
    for j=1:size(b_base,3)      %cell
        temp=squeeze(b_base(:,:,j)).*base_design(i-1:i+5,:);
        filt_out_base(i,j)=sum(temp(:));
        temp=squeeze(b_bias(:,:,j)).*base_design(i-1:i+5,:);
        filt_out_basex(i,j)=sum(temp(:));
        temp=sum(squeeze(b_bias(:,:,j)).*bias_design(i-1:i+5,:));
        filt_out_bias(i,j)=sum(temp(:));
        temp=sum(squeeze(b_base(:,:,j)).*bias_design(i-1:i+5,:));
        filt_out_biasx(i,j)=sum(temp(:));
    end
end

filt_out_base=filt_out_base(2:end,:);
filt_out_bias=filt_out_bias(2:end,:);
resp_out_base=resp_raw_base_binned2(2:end-5,:);
resp_out_bias=resp_raw_bias_binned2(2:end-5,:);

for i=1:size(filt_out_base,2)
    temp=filt_out_base(:,i);
    bins=(min(temp):(max(temp)-min(temp))/50:max(temp));
    temp2=filt_out_bias(:,i);
    bins2=(min(temp2):(max(temp2)-min(temp2))/50:max(temp2));
    for j=1:length(bins)-1
        nl_in_base(i,j)=mean([bins(j) bins(j+1)]);
        nl_out_base(i,j)=mean(resp_out_base(find(temp>bins(j) & temp<bins(j+1)),i));
        nl_in_bias(i,j)=mean([bins2(j) bins2(j+1)]);
        nl_out_bias(i,j)=mean(resp_out_bias(find(temp2>bins2(j) & temp2<bins2(j+1)),i));  %Benucci use the same N: pre and post
    end
end

for i=1:size(filt_out_base,2)       %cells (bins)
    for j=1:size(filt_out_base,1)
        aa=find(nl_in_base(i,:)>filt_out_base(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        model_pred_base(j,i)=nl_out_base(i,aa);

        aa=find(nl_in_bias(i,:)>filt_out_basex(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        model_pred_basex(j,i)=nl_out_bias(i,aa);

        aa=find(nl_in_bias(i,:)>filt_out_bias(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        model_pred_bias(j,i)=nl_out_bias(i,aa);

        aa=find(nl_in_base(i,:)>filt_out_biasx(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        model_pred_biasx(j,i)=nl_out_base(i,aa);
    end
end

%% normalize predictions
for i=1:9
    model_pred_base_norm(i,:)=model_pred_base(:,i)./norm_fact(i);
    model_pred_basex_norm(i,:)=model_pred_basex(:,i)./norm_fact(i);
    model_pred_bias_norm(i,:)=model_pred_bias(:,i)./norm_fact(i);
    model_pred_biasx_norm(i,:)=model_pred_biasx(:,i)./norm_fact(i);
end

% tuning of model
base_design2=base_design(2:end-5,:);
bias_design2=bias_design(2:end-5,:);
for i=1:9
    for j=1:10
        temp=find(base_design2(:,j)==1);
        model_pred_base_binned(i,j)=nanmean(model_pred_base(temp,i));
        model_pred_basex_binned(i,j)=nanmean(model_pred_basex(temp,i));
        temp=find(bias_design2(:,j)==1);
        model_pred_bias_binned(i,j)=nanmean(model_pred_bias(temp,i));
        model_pred_biasx_binned(i,j)=nanmean(model_pred_biasx(temp,i));       
    end
end

for i=1:size(resp_base_binned,1)
    for j=1:size(resp_base_binned,1)
        corr_base(i,j)=akcorrcoef(resp_base_binned(i,:),resp_base_binned(j,:));
        corr_bias(i,j)=akcorrcoef(resp_bias_binned(i,:),resp_bias_binned(j,:));

        corr_model_base(i,j)=akcorrcoef(model_pred_base_binned(i,:),model_pred_base_binned(j,:));
        corr_model_bias(i,j)=akcorrcoef(model_pred_bias_binned(i,:),model_pred_bias_binned(j,:));
        corr_model_basex(i,j)=akcorrcoef(model_pred_basex_binned(i,:),model_pred_basex_binned(j,:));
        corr_model_biasx(i,j)=akcorrcoef(model_pred_biasx_binned(i,:),model_pred_biasx_binned(j,:));
    end
end

%% simpler plots--data

figure

subplot(2,2,1)
temp=circshift(base_count(1:end-1),[0 4]);
bar((0:20:180),[temp base_count(end)]/15000,1);
hold on;box off
plot((-10:10:190),(1/10)*ones(21,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')

subplot(2,2,2)
errorline((0:20:160),mean(resp_base_norm,2),std(resp_base_norm,0,2)/sqrt(size(resp_base_norm,2)),'k')
hold on; box off
plot((0:20:160),mean(resp_base_norm,2),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(nanmean(model_pred_base_norm,2),[4 0]),'b','LineWidth',2)
plot((0:20:160),circshift(nanmean(model_pred_basex_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),ones(9,1),':r')

subplot(2,2,3)
temp=circshift(bias_count(1:end-1),[0 4]);
bar((0:20:180),[temp bias_count(end)]/15000,1);
hold on;box off
plot((-10:10:190),(1/10)*ones(21,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')
xlabel('Stim')

subplot(2,2,4)
errorline((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),circshift(std(resp_bias_norm,0,2)/sqrt(size(resp_bias_norm,2)),[4 0]),'k')
hold on; box off
plot((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(nanmean(model_pred_bias_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),circshift(nanmean(model_pred_biasx_norm,2),[4 0]),'b','LineWidth',2)
plot((0:20:160),ones(9,1),':r')


%% plots
figure

%normalized base
subplot(4,2,1)
plot(base(100:120),(1:21),'.k')
hold on;box off
axis([-5 185 0 21])
set(gca,'PlotBoxAspectRatio',[2 5 1])

subplot(4,2,3)
temp=circshift(base_count(1:end-1),[0 4]);
bar((0:20:180),[temp base_count(end)]);
axis([-5 185 0 1.2*max(base_count)])
hold on;box off
plot((0:20:180),(length(base)/length(ori_base))*ones(10,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')

subplot(4,2,2)
imagesc(circshift(resp_base_norm(:,100:120)',[0 4]))
set(gca,'PlotBoxAspectRatio',[2 5 1])
box off

subplot(4,2,4)
errorline((0:20:160),mean(resp_base_norm,2),std(resp_base_norm,0,2)/sqrt(size(resp_base_norm,2)),'k')
hold on; box off
plot((0:20:160),mean(resp_base_norm,2),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(mean(pred_resp_basex_norm,2),[4 0]),'r','LineWidth',2)

% biased
subplot(4,2,5)
plot_stim=bias(100:120);
plot_stim(plot_stim==180)=0;  %blank is now 180
plot_stim(plot_stim==200)=180;  %blank is now 180
plot(plot_stim,(1:21),'.k')
hold on;box off
axis([-5 185 0 21])
set(gca,'PlotBoxAspectRatio',[2 5 1])

subplot(4,2,7)
temp=circshift(bias_count(1:end-1),[0 4]);
bar((0:20:180),[temp bias_count(end)]);
hold on;box off
plot((0:20:180),(length(bias)/length(ori_base))*ones(10,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')
xlabel('Stim')

subplot(4,2,6)
imagesc(circshift(resp_bias_norm(:,100:120)',[0 4]))
set(gca,'PlotBoxAspectRatio',[2 5 1])
box off

subplot(4,2,8)
errorline((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),circshift(std(resp_bias_norm,0,2)/sqrt(size(resp_bias_norm,2)),[4 0]),'k')
plot((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
hold on; box off
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(mean(pred_resp_biasx_norm,2),[4 0]),'r','LineWidth',2)

%% simpler plots

figure

subplot(2,2,1)
temp=circshift(base_count(1:end-1),[0 4]);
bar((0:20:180),[temp base_count(end)]/15000,1);
hold on;box off
plot((-10:10:190),(1/10)*ones(21,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')

subplot(2,2,2)
errorline((0:20:160),mean(resp_base_norm,2),std(resp_base_norm,0,2)/sqrt(size(resp_base_norm,2)),'k')
hold on; box off
plot((0:20:160),mean(resp_base_norm,2),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(mean(pred_resp_basex_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),ones(9,1),':r')

subplot(2,2,3)
temp=circshift(bias_count(1:end-1),[0 4]);
bar((0:20:180),[temp bias_count(end)]/15000,1);
hold on;box off
plot((-10:10:190),(1/10)*ones(21,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')
xlabel('Stim')

subplot(2,2,4)
errorline((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),circshift(std(resp_bias_norm,0,2)/sqrt(size(resp_bias_norm,2)),[4 0]),'k')
hold on; box off
plot((0:20:160),circshift(mean(resp_bias_norm,2),[4 0]),'ok','MarkerFaceColor','k')
ylabel('Mean response')
xlabel('Ori preference')
axis([-10 180 0.8 1.2])
plot((0:20:160),circshift(mean(pred_resp_biasx_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),ones(9,1),':r')

stop
% save as *_process
%% like Benucci et al 1 e,f, i, j

figure

subplot(3,4,1)
plot_stim=base(100:120);
plot_stim(plot_stim==200)=180;  %blank is now 180
plot(plot_stim,(1:21),'.k')
hold on;box off
axis([-5 185 0 21])
set(gca,'PlotBoxAspectRatio',[2 5 1])

clear base_count
subplot(3,4,2)
for i=1:length(ori_base)
    base_count(i)=length(find(base==ori_base(i)));
end
bar((0:20:180),base_count);
axis([-5 185 0 1.2*max(base_count)])
hold on;box off
plot((0:20:180),(length(base)/length(ori_base))*ones(10,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')

subplot(3,4,3)
plot_resp=resp_raw_binned(:,100:120);
imagesc(plot_resp')
set(gca,'PlotBoxAspectRatio',[2 5 1])
box off

subplot(3,4,4)
bar((0:20:160),mean(resp_raw_binned,2))
ylabel('Mean response')
xlabel('Ori preference')
hold on; box off
axis([-10 180 0 10])

%normalized base
subplot(3,4,5)
plot(plot_stim,(1:21),'.k')
hold on;box off
axis([-5 185 0 21])
set(gca,'PlotBoxAspectRatio',[2 5 1])

subplot(3,4,6)
bar((0:20:180),base_count);
axis([-5 185 0 1.2*max(base_count)])
hold on;box off
plot((0:20:180),(length(base)/length(ori_base))*ones(10,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')

subplot(3,4,7)
norm_fact=mean(resp_raw_binned,2);
for i=1:9
    resp_raw_norm(i,:)=resp_raw_binned(i,:)./norm_fact(i,:);
end
imagesc(resp_raw_norm(:,100:120)')
set(gca,'PlotBoxAspectRatio',[2 5 1])
box off

subplot(3,4,8)
bar((0:20:160),mean(resp_raw_norm,2))
ylabel('Mean response')
xlabel('Ori preference')
hold on; box off
axis([-10 180 0 1.5])

% biased
subplot(3,4,9)
plot_stim=bias(100:120);
plot_stim(plot_stim==180)=0;  %blank is now 180
plot_stim(plot_stim==200)=180;  %blank is now 180
plot(plot_stim,(1:21),'.k')
hold on;box off
axis([-5 185 0 21])
set(gca,'PlotBoxAspectRatio',[2 5 1])

subplot(3,4,10)
for i=1:length(ori_bias)
    base_count(i)=length(find(bias==ori_bias(i)));
end
base_count(1)=base_count(1)+base_count(end-1);
base_count=[base_count(1:9) base_count(end)];
bar((0:20:180),base_count);
hold on;box off
plot((0:20:180),(length(base)/length(ori_base))*ones(10,1),':k')
set(gca,'XLim',[-10 185])
ylabel('Presentation num')
xlabel('Stim')

subplot(3,4,11)
for i=1:9
    resp_binned_norm(i,:)=resp_raw_binned2(i,:)./norm_fact(i,:);
end
imagesc(resp_binned_norm(:,100:120)')
set(gca,'PlotBoxAspectRatio',[2 5 1])
box off

subplot(3,4,12)
bar((0:20:160),mean(resp_binned_norm,2))
ylabel('Mean response')
xlabel('Ori preference')
hold on; box off
axis([-10 180 0 1.5])

%% basic tuning plots

% figure
% for i=1:size(resp_base2,1)
%     plot(resp_base2(i,:),'.k-')
%     hold on
%     plot(resp_bias2(i,:),'.r-')
%     pause
%     clf
% end
stop
% save as *_process