clear
% ACUTE/Anesthetized files

for n=28:33
    clearvars -except n
    if n==1
        load('129r001p173_entropy')
        name='129r001p173_corr';
    elseif n==2
        load('130l001p169_entropy')
        name='130l001p169_corr';
    elseif n==3
        load('140l001p107_entropy')
        name='140l001p107_corr';
    elseif n==4
        load('140l001p122_entropy')
        name='140l001p122_corr';
    elseif n==5
        load('140r001p105_entropy')
        name='140r001p105_corr';
    elseif n==6
        load('140r001p122_entropy')
        name='140r001p122_corr';
    elseif n==7
        load('130l001p170_entropy')
        name='130l001p170_corr';
    elseif n==8
        load('140l001p108_entropy')
        name='140l001p108_corr';
    elseif n==9
        load('140l001p110_entropy')
        name='140l001p110_corr';
    elseif n==10
        load('140r001p107_entropy')
        name='140r001p107_corr';
    elseif n==11
        load('140r001p109_entropy')
        name='140r001p109_corr';
    elseif n==12
        load('lowcon114_entropy')
        name='lowcon114_corr';
    elseif n==13
        load('lowcon115_entropy')
        name='lowcon115_corr';
    elseif n==14
        load('lowcon116_entropy')
        name='lowcon116_corr';
    elseif n==15
        load('lowcon117_entropy')
        name='lowcon117_corr';
    elseif n==16
        load('140l113_awaketime_entropy')
        name='140l113_awaketime_corr';
    elseif n==17
        load('140r113_awaketime_entropy')
        name='140r113_awaketime_corr';
        
    elseif n==18 % start of experiment 141 files
        load('141r001p006_awaketime_entropy')
        name='141r001p006_awaketime_corr';
    elseif n==19
        load('141r001p007_awaketime6_entropy')
        name='141r001p007_awaketime6_corr';
    elseif n==20
        load('141r001p009_awaketime_fine_entropy')
        name='141r001p009_awaketime_fine_corr';
    elseif n==21 % rotated AT 4:1 (80°)
        load('141r001p024_awaketime_entropy')
        name='141r001p024_awaketime_corr';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_entropy')
        name='141r001p025_awaketime6_corr';
    elseif n==23 % rotated AT fineori (90°??)
        load('141r001p027_awaketime_fine_entropy')
        name='141r001p027_awaketime_fine_corr';
    elseif n==24 % rotated fineori (40°)
        load('141r001p038_awaketime_fine_entropy')
        name='141r001p038_awaketime_fine_corr';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_entropy')
        name='141r001p039_awaketime6_corr';
    elseif n==26 % rotated awaketime 4:1 (120°)
        load('141r001p041_awaketime_entropy')
        name='141r001p041_awaketime_corr';
    elseif n==27
        load('141r001p114_entropy')
        name='141r001p114_corr';
        
    elseif n==28    % start of 142 files
        load('142l001p002_awaketime_entropy')
        name='142l001p002_awaketime_corr';
    elseif n==29
        load('142l001p004_awaketime_fine_entropy')
        name='142l001p004_awaketime_fine_corr';
    elseif n==30
        load('142l001p006_awaketime6_entropy')
        name='142l001p006_awaketime6_corr';
    elseif n==31
        load('142l001p007_awaketime_entropy')
        name='142l001p007_awaketime_corr';
    elseif n==32
        load('142l001p009_awaketime_fine_entropy')
        name='142l001p009_awaketime_fine_corr';
    elseif n==33
        load('142l001p010_awaketime6_entropy')
        name='142l001p010_awaketime6_corr';
    end
%     _entropy files already exclude non-responsive units
    %% reorganize so adapting ori is in the middle
    [Y,I]=sort(oripref);
    val_mode=mode(bias); %should be 0 or 80
    x=val_mode+90;
    tmp=abs(Y-x);
    tmp=find(tmp==min(tmp));
    Y2=[Y(tmp(1):end) Y(1:tmp(1))];
    I2=[I(tmp(1):end) I(1:tmp(1)-1)];
    
    %%   subsampling
%     keep=find(base==val_mode); % find trials of biased ori in base environment
%     xx=length(keep);    % # of trials
%     rat=length(find(bias==val_mode))/length(find(bias==20)); % bias ratio
%     for i=1:length(ori_base)-1
%         tmp=find(base==ori_base(i)); % find trials of ori i in uniform
%         if isempty(tmp)
%             continue
%         elseif ori_base(i)==val_mode
%             continue
%         end
%         tmp2=randperm(length(tmp)); % randomized trial indeces of tmp
%         tmp=tmp(tmp2(1:round(xx/rat))); % takes subsample from tmp2 proportional to bias ratio
%         keep=[keep(:);tmp(:)];  % stores the indeces of the sampled trials
%     end
%     keep=sort(keep);
%     keep2=randperm(length(bias));
%     keep2=keep2(1:length(keep));
    
    aa2=resp_raw_base(I2,:);    % raw responses straight out of pre-process
    bb2=resp_raw_bias(I2,:);
    cc=resp_base(I2,:);
    dd=resp_bias(I2,:);
    ee=resp_raw_base(I2,keep);  % subsampled responses
    ff=resp_raw_bias(I2,keep2);
    clear tmp*
    %% separate noise correlations
    % spike counts to individual trials of each stim
    stim_resp_u=cell(size(aa2,1),length(ori_base)-1);    % all trials
    stim_resp_b=cell(size(aa2,1),length(ori_base)-1);
    stim_resp_u1=cell(size(aa2,1),length(ori_base)-1);   % first half of trials
    stim_resp_b1=cell(size(aa2,1),length(ori_base)-1);
    stim_resp_u2=cell(size(aa2,1),length(ori_base)-1);   % second half of trials
    stim_resp_b2=cell(size(aa2,1),length(ori_base)-1);
    z=round(length(base)/2);    % half # of trials
    for i=1:size(aa2,1)
        tmp=zscore(aa2(i,:));
        tmp2=zscore(bb2(i,:));
        
        for j=1:length(ori_base)-1
            [~,x]=find(base==ori_base(j));
            [~,y]=find(bias==ori_base(j));
            if isempty(x)
                stim_resp_u{i,j}=nan;   % all trials
                stim_resp_u1{i,j}=nan;  % first half of trials
                stim_resp_u2{i,j}=nan;  % second half of trials
            else
                stim_resp_u{i,j}=tmp(x);
                stim_resp_u1{i,j}=tmp(x(x<=z));
                stim_resp_u2{i,j}=tmp(x(x>z));
            end
            if isempty(y)
                stim_resp_b{i,j}=nan;
                stim_resp_b1{i,j}=nan;
                stim_resp_b2{i,j}=nan;
            else
                stim_resp_b{i,j}=tmp2(y);
                stim_resp_b1{i,j}=tmp2(y(y<=z));
                stim_resp_b2{i,j}=tmp2(y(y>z));
            end
        end
    end
    clear x y z
    %% conditional variance
    % var(R1 | R2)=wR2^2+sigma^2
    bins=0:1/6:1;
    CondVar_u=cell(size(aa2,1),size(aa2,1));
    CondVar_b=cell(size(aa2,1),size(aa2,1));
    for i = 1:size(aa2,1)
        for j=1:size(aa2,1)
            % normalized response vector for 2 units
            tmp1=(aa2(i,:)./max(aa2(i,:)));
            tmp2=(aa2(j,:)./max(aa2(j,:)));
            tmp3=(bb2(i,:)./max(bb2(i,:)));
            tmp4=(bb2(j,:)./max(bb2(j,:)));
            cv1=histcounts2(tmp1,tmp2,bins,bins);
            cv2=histcounts2(tmp3,tmp4,bins,bins);
            for k=1:length(cv1)
                cv1(:,k)=cv1(:,k)./max(cv1(:,k));
                cv2(:,k)=cv2(:,k)./max(cv2(:,k));
            end
            CondVar_u{i,j}=cv1;
            CondVar_b{i,j}=cv2;
        end
    end
    %% calculate correlations (1: all responses; 2: Signal; 3: noise)
    corr_base_measured=nan*zeros(size(aa2,1),size(aa2,1));
    corr_bias_measured=nan*zeros(size(aa2,1),size(aa2,1));
    corr_base_sub=nan*zeros(size(aa2,1),size(aa2,1));
    corr_bias_sub=nan*zeros(size(aa2,1),size(aa2,1));
    sig_base=nan*zeros(size(aa2,1),size(aa2,1));
    sig_bias=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base1=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias1=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base2=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias2=nan*zeros(size(aa2,1),size(aa2,1));
    for i=1:size(aa2,1)-1
        for j=i+1:size(aa2,1)
            % correlations on all responses (noise+signal):
            corr_base_measured(i,j)=akcorrcoef(aa2(i,:)',aa2(j,:)');  % measured from full data set
            corr_bias_measured(i,j)=akcorrcoef(bb2(i,:)',bb2(j,:)');
            corr_base_sub(i,j)=akcorrcoef(ee(i,:)',ee(j,:)');       % measured from subsampled matched statistics
            corr_bias_sub(i,j)=akcorrcoef(ff(i,:)',ff(j,:)');
            
            % signal correlations: tuning curves
            sig_base(i,j)=akcorrcoef(cc(i,:)',cc(j,:)');
            sig_bias(i,j)=akcorrcoef(dd(i,:)',dd(j,:)');
            
            % noise/spike count correlations
            tmp=nan*zeros(1,length(ori_base)-1);
            tmp2=nan*zeros(1,length(ori_base)-1);
            tmp3=nan*zeros(1,length(ori_base)-1);
            tmp4=nan*zeros(1,length(ori_base)-1);
            tmp5=nan*zeros(1,length(ori_base)-1);
            tmp6=nan*zeros(1,length(ori_base)-1);
            for k = 1:length(ori_base)-1
                [~,sd]=find(stim_resp_u{i,k}>=3);
                [~,sd2]=find(stim_resp_u{j,k}>=3);
                x=setdiff(1:length(stim_resp_u{i,k}),[sd sd2]);
                tmp(k) =akcorrcoef(stim_resp_u{i,k}(x),stim_resp_u{j,k}(x));
                
                [~,sd]=find(stim_resp_b{i,k}>=3);
                [~,sd2]=find(stim_resp_b{j,k}>=3);
                x=setdiff(1:length(stim_resp_b{i,k}),[sd sd2]);
                tmp2(k)=akcorrcoef(stim_resp_b{i,k}(x),stim_resp_b{j,k}(x));
                
                [~,sd]=find(stim_resp_u1{i,k}>=3);
                [~,sd2]=find(stim_resp_u1{j,k}>=3);
                x=setdiff(1:length(stim_resp_u1{i,k}),[sd sd2]);
                tmp3(k)=akcorrcoef(stim_resp_u1{i,k}(x),stim_resp_u1{j,k}(x));
                
                [~,sd]=find(stim_resp_b1{i,k}>=3);
                [~,sd2]=find(stim_resp_b1{j,k}>=3);
                x=setdiff(1:length(stim_resp_b1{i,k}),[sd sd2]);
                tmp4(k)=akcorrcoef(stim_resp_b1{i,k}(x),stim_resp_b1{j,k}(x));
                
                [~,sd]=find(stim_resp_u2{i,k}>=3);
                [~,sd2]=find(stim_resp_u2{j,k}>=3);
                x=setdiff(1:length(stim_resp_u2{i,k}),[sd sd2]);
                tmp5(k)=akcorrcoef(stim_resp_u2{i,k}(x),stim_resp_u2{j,k}(x));
                
                [~,sd]=find(stim_resp_b2{i,k}>=3);
                [~,sd2]=find(stim_resp_b2{j,k}>=3);
                x=setdiff(1:length(stim_resp_b2{i,k}),[sd sd2]);
                tmp6(k)=akcorrcoef(stim_resp_b2{i,k}(x),stim_resp_b2{j,k}(x));
            end
            noise_base(i,j)=mean(tmp);
            noise_bias(i,j)=mean(tmp2);
            noise_base1(i,j)=mean(tmp3);
            noise_bias1(i,j)=mean(tmp4);
            noise_base2(i,j)=mean(tmp5);
            noise_bias2(i,j)=mean(tmp6);
        end
    end
    
    %%   plot smoothed measured effect vs prediction from tuning
    % for smoother plots
    aa=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
    aa=aa/sum(aa(:));
    
%     figure
%     supertitle('response correlations (benucci acute)')
%     subplot(231)
%     imagesc(conv2(corr_base_measured,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform measured')
%     subplot(232)
%     imagesc(conv2(corr_bias_measured,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased measured')
%     subplot(233)
%     imagesc(conv2(corr_bias_measured-corr_base_measured,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
%     subplot(234)
%     imagesc(conv2(corr_base_sub,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform with subsampled bias')
%     subplot(235)
%     imagesc(conv2(corr_bias_sub,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased with subsampled bias')
%     subplot(236)
%     imagesc(conv2(corr_bias_sub-corr_base_sub,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Bias-uniform/bias')
%     
%     figure
%     supertitle('signal correlations (benucci acute)')
%     subplot(131)
%     imagesc(conv2(sig_base,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform')
%     subplot(132)
%     imagesc(conv2(sig_bias,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased')
%     subplot(133)
%     imagesc(conv2(sig_bias-sig_base,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
    
%     figure
%     supertitle('noise correlations (benucci acute)')
%     subplot(131)
%     imagesc(conv2(noise_base,aa,'same'),[-0.2 0.2])
%     axis square;box off
%     title('Uniform')
%     subplot(132)
%     imagesc(conv2(noise_bias,aa,'same'),[-0.2 0.2])
%     axis square;box off
%     title('Biased')
%     subplot(133)
%     imagesc(conv2(noise_bias-noise_base,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
    
    % % % % noise correlations, first vs second half of trials
%     figure
%     supertitle('Noise correlations half trials (acute 2:1)')
%     subplot(331)
%     imagesc(conv2(noise_base1,aa,'same'),[-0.25 0.25])
%     axis square;box off
%     title('Uniform 1st')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(332)
%     imagesc(conv2(noise_base2,aa,'same'),[-0.25 0.25])
%     axis square;box off
%     title('Uniform 2nd')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(333)
%     imagesc(conv2(noise_base2-noise_base1,aa,'same'),[-0.035 0.035])
%     axis square;box off
%     title('Diff: uniform (2-1)')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(334)
%     imagesc(conv2(noise_bias1,aa,'same'),[-0.25 0.25])
%     axis square;box off
%     title('Bias 1st')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(335)
%     imagesc(conv2(noise_bias2,aa,'same'),[-0.25 0.25])
%     axis square;box off
%     title('Bias 2nd')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(336)
%     imagesc(conv2(noise_bias2-noise_bias1,aa,'same'),[-0.035 0.035])
%     axis square;box off
%     title('Diff: Bias (2-1)')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(337)
%     imagesc(conv2(noise_bias1-noise_base1,aa,'same'),[-0.055 0.055])
%     axis square;box off
%     title('Diff: Bias-Uniform (first half)')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})
%     subplot(338)
%     imagesc(conv2(noise_bias2-noise_base2,aa,'same'),[-0.055 0.055])
%     axis square;box off
%     title('Diff: Bias-Uniform (second half)')
%     set(gca,'TickDir','out','XTick',1:2:11,'XTickLabel',{'-90','-60','-30','0','30','60'},...
%         'YTick',1:2:11,'YTickLabel',{'-90','-60','-30','0','30','60'})

    clear tmp* i j k x y ans cv1 cv2
    save(name);
end
stop

%% shuffle trials: contribution of signal v noise

% %   shuffle trials for each channel/unit
% for i=1:size(aa2,1)
%     shuf=randperm(length(aa2));
%     shuf2=randperm(length(ee));
%     aas(i,:)=aa2(i,shuf);
%     bbs(i,:)=bb2(i,shuf);
%     ees(i,:)=ee(i,shuf2);
%     ffs(i,:)=ff(i,shuf2);
% end
% %   calculate shuffled correlations
% for i=1:size(aa2,1)
%     for j=1:size(aa2,1)
%         corr_base_shuf(i,j)=akcorrcoef(aas(i,:),aas(j,:));
%         corr_bias_shuf(i,j)=akcorrcoef(bbs(i,:),bbs(j,:));
%         corr_base_sub_shuf(i,j)=akcorrcoef(ees(i,:),ees(j,:));
%         corr_bias_sub_shuf(i,j)=akcorrcoef(ffs(i,:),ffs(j,:));
%     end
% end
% %   plot smoothed measured effect vs prediction from tuning
% figure
% supertitle('shuffled correlations (benucci acute)')
% subplot(231)
% imagesc(conv2(corr_base_shuf,aa,'same'),[-0.03 0.35])
% axis square;box off
% title('Uniform measured')
% subplot(232)
% imagesc(conv2(corr_bias_shuf,aa,'same'),[-0.03 0.35])
% axis square;box off
% title('Biased measured')
% subplot(233)
% imagesc(conv2(corr_bias_shuf-corr_base_shuf,aa,'same'),[-0.05 0.05])
% axis square;box off
% title('Biased-uniform')
% subplot(234)
% imagesc(conv2(corr_base_sub_shuf,aa,'same'),[-0.03 0.35])
% axis square;box off
% title('Uniform with subsampled bias')
% subplot(235)
% imagesc(conv2(corr_bias_sub_shuf,aa,'same'),[-0.03 0.35])
% axis square;box off
% title('Biased with subsampled bias')
% subplot(236)
% imagesc(conv2(corr_bias_sub_shuf-corr_base_sub_shuf,aa,'same'),[-0.05 0.05])
% axis square;box off
% title('Bias-uniform/bias')
% %   calculate difference between measured and shuffled correlations
% r_u_diff=corr_base_measured-corr_base_shuf;
% r_b_diff=corr_bias_measured-corr_bias_shuf;
% r_u_sub_diff=corr_base_sub-corr_base_sub_shuf;
% r_b_sub_diff=corr_bias_sub-corr_bias_sub_shuf;
% %   plot difference between measured and shuffled correlations
% figure
% supertitle('diff(measured-shuffled) correlations (benucci acute)')
% subplot(231)
% imagesc(conv2(r_u_diff,aa,'same'),[-0.2 0.75])
% axis square;box off
% title('Uniform measured')
% subplot(232)
% imagesc(conv2(r_b_diff,aa,'same'),[-0.2 0.75])
% axis square;box off
% title('Biased measured')
% subplot(233)
% imagesc(conv2(r_b_diff-r_u_diff,aa,'same'),[-0.05 0.075])
% axis square;box off
% title('Biased-uniform')
% subplot(234)
% imagesc(conv2(r_u_sub_diff,aa,'same'),[-0.2 0.75])
% axis square;box off
% title('Uniform with subsampled bias')
% subplot(235)
% imagesc(conv2(r_b_sub_diff,aa,'same'),[-0.2 0.75])
% axis square;box off
% title('Biased with subsampled bias')
% subplot(236)
% imagesc(conv2(r_b_sub_diff-r_u_sub_diff,aa,'same'),[-0.05 0.075])
% axis square;box off
% title('Bias-uniform/bias')

clear i j tmp* temp* xx u_xx b_xx
stopacute_SAVE

%% benucci awake (each file calculated separately)
clear
tmp1_21=cell(12,12);    % measured correlations:
tmp2_21=cell(12,12);
tmp1_61=cell(12,12);
tmp2_61=cell(12,12);
tmp1_21_sub=cell(12,12);% trial sub-sampled correlations:
tmp2_21_sub=cell(12,12);
tmp1_61_sub=cell(12,12);
tmp2_61_sub=cell(12,12);
tmp1sig_21=cell(12,12);    % measured correlations:
tmp2sig_21=cell(12,12);
tmp1sig_61=cell(12,12);
tmp2sig_61=cell(12,12);
tmp1noise_21=cell(12,12);    % measured correlations:
tmp2noise_21=cell(12,12);
tmp1noise_61=cell(12,12);
tmp2noise_61=cell(12,12);
% tmp1s_21=cell(12,12);   % shuffled correlations:
% tmp2s_21=cell(12,12);
% tmp1s_61=cell(12,12);
% tmp2s_61=cell(12,12);
tmp1noise1_61=cell(12,12);
tmp1noise2_61=cell(12,12);
tmp2noise1_61=cell(12,12);
tmp2noise2_61=cell(12,12);
tmp1noise1_21=cell(12,12);
tmp1noise2_21=cell(12,12);
tmp2noise1_21=cell(12,12);
tmp2noise2_21=cell(12,12);
tmp_dyn1=cell(12,12);
tmp_dyn2=cell(12,12);
tmp_dyn3=cell(12,12);
tmp_dyn4=cell(12,12);
temp_dyn1=cell(12,12);
temp_dyn2=cell(12,12);
temp_dyn3=cell(12,12);
temp_dyn4=cell(12,12);
for a=[17 18 20 22 25 27] %15:27
    clearvars -except tmp1_21 tmp2_21 tmp1_61 tmp2_61 a tmp1_21_sub tmp2_21_sub...
         tmp1_61_sub tmp2_61_sub tmp1sig_21 tmp2sig_21 tmp1sig_61 tmp2sig_61...
         tmp1noise* tmp2noise* tmp_dyn* temp_dyn*

    if a==1
        load('cadetv1p194_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==2
%         load('cadetv1p195_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==3
%         load('cadetv1p245_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==4
%         load('cadetv1p246_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==5
%         load('cadetv1p345_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==6
%         load('cadetv1p346_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==7
%         load('cadetv1p347_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==8
%         load('cadetv1p348_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==9
%         load('cadetv1p349_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==10
%         load('cadetv1p350_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==11
%         load('cadetv1p351_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==12
%         load('cadetv1p352_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==13
%         load('cadetv1p353_entropy','filename','ori*','tune*','resp*','keep*');
%     elseif a==14
%         load('cadetv1p355_entropy','filename','ori*','tune*','resp*','keep*');
% %         load('cadetv1p405_entropy','filename','ori*','tune*','resp*','keep*');
% %         load('cadetv1p418_entropy','filename','ori*','tune*','resp*','keep*');
    elseif a==15
%         load('cadetv1p366_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p366_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==16
%         load('cadetv1p371_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p371_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==17
%         load('cadetv1p384_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p384_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==18
%         load('cadetv1p385_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p385_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==19
%         load('cadetv1p392_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p392_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==20
%         load('cadetv1p403_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p403_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==21
%         load('cadetv1p419_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p419_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==22
% % % %         load('cadetv1p404_entropy','filename','ori*','tune*','resp*','spont','keep*');
%         load('cadetv1p432_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p432_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==23
        continue
%         load('cadetv1p437_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
%         load('cadetv1p437_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==24
%         load('cadetv1p438_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p438_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==25
%         load('cadetv1p460_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p460_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==26
%         load('cadetv1p467_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p467_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==27
%         load('cadetv1p468_entropy_drop','filename','ori*','tune*','resp*','spont','keep*');
        load('cadetv1p468_entropy','filename','ori*','tune*','resp*','spont','keep*');
    elseif a==28
        load('cadetv1p422_entropy','filename','ori*','tune*','resp*','spont','keep*');
    end
    % only includes responsive units from _entropy analysis
    %% order by orientation preference 0:180
    try
        [Y,I]=sort(oripref);
    catch
        [Y,I]=sort(oripref_u);
    end
    % reorganize so 0/180 in the middle
    tmp=abs(Y-90);
    tmp=find(tmp==min(tmp));
    Y2=[Y(tmp(1):end) Y(1:tmp(1))];
    I2=[I(tmp(1):end) I(1:tmp(1)-1)];
    %raw responses to each environment
    try
        ob2=oribias(I2);
    catch
        ob2=oribias_u(I2);
    end
    
    aa2=resp_uniform(I2,:);    % raw responses straight out of pre-process
    bb2=resp_bias(I2,:);
    cc=tune_u(I2,:);
    dd=tune_b(I2,:);
    ee=resp_uniform(I2,keep);  % subsampled responses
    ff=resp_bias(I2,keep2);
    spont=spont(I2,:);
    clear tmp tmp2
    
    %%  subsampling
%     keep=find(oris_u==0);   % find trials of biased ori in uniform environment
%     xx=length(keep);        % # of trials
%     rat=length(find(oris_b==0))/length(find(oris_b==20)); % approximate bias ratio
%     for i=2:size(tune_u,2)         % loop for each stim ori
%         tmp=find(oris_u==(i-1)*20);     % find trials of ori i in uniform
%         if isempty(tmp)
%             continue
%         end
%         tmp2=randperm(length(tmp));     % randomized trial indeces of tmp
%         tmp=tmp(tmp2(1:round(xx/rat))); % takes subsample from tmp2 proportional to bias ratio
%         keep=[keep(:);tmp(:)];          % stores the indeces of the sampled trials
%     end
%     keep=sort(keep);
%     keep2=randperm(size(resp_bias,2));
%     keep2=keep2(1:length(keep));
        
    %% separate noise correlations
    % spike counts to individual trials of each stim
    stim_resp_u=cell(size(aa2,1),9);
    stim_resp_b=cell(size(aa2,1),9);
    stim_resp_u1=cell(size(aa2,1),9);   % first half of trials
    stim_resp_b1=cell(size(aa2,1),9);
    stim_resp_u2=cell(size(aa2,1),9);   % second half of trials
    stim_resp_b2=cell(size(aa2,1),9);
    z=round(length(oris_u)/2);      % uniform half trial length
    z2=round(length(oris_b)/2);     % bias half trial length
    for i=1:size(aa2,1)
        tmp=zscore(aa2(i,:));
        tmp2=zscore(bb2(i,:));
        
        for j=1:length(oris)
            [~,x]=find(oris_u==oris(j));
            [~,y]=find(oris_b==oris(j));
            if isempty(x)
                stim_resp_u{i,j}=nan;   % all trials
                stim_resp_u1{i,j}=nan;  % first half of trials
                stim_resp_u2{i,j}=nan;  % second half of trials
            else
                stim_resp_u{i,j}=tmp(x);
                stim_resp_u1{i,j}=tmp(x(x<=z));
                stim_resp_u2{i,j}=tmp(x(x>z));
            end
            if isempty(y)
                stim_resp_b{i,j}=nan;
                stim_resp_b1{i,j}=nan;
                stim_resp_b2{i,j}=nan;
            else
                stim_resp_b{i,j}=tmp2(y);
                stim_resp_b1{i,j}=tmp2(y(y<=z2));
                stim_resp_b2{i,j}=tmp2(y(y>z2));
            end
        end
    end
    clear x y z z2 tmp tmp2
    %% conditional variance
    % var(R1 | R2)=wR2^2+sigma^2
    bins=0:1/6:1;
    CondVar_u=cell(size(aa2,1),size(aa2,1));
    CondVar_b=cell(size(aa2,1),size(aa2,1));
    for i = 1:size(aa2,1)
        for j=1:size(aa2,1)
            % normalized response vector for 2 units
            tmp1=(aa2(i,:)./max(aa2(i,:)));
            tmp2=(aa2(j,:)./max(aa2(j,:)));
            tmp3=(bb2(i,:)./max(bb2(i,:)));
            tmp4=(bb2(j,:)./max(bb2(j,:)));
            cv1=histcounts2(tmp1,tmp2,bins,bins);
            cv2=histcounts2(tmp3,tmp4,bins,bins);
            for k=1:length(cv1)
                cv1(:,k)=cv1(:,k)./max(cv1(:,k));
                cv2(:,k)=cv2(:,k)./max(cv2(:,k));
            end
            CondVar_u{i,j}=cv1;
            CondVar_b{i,j}=cv2;
        end
    end
    clear tmp tmp1 tmp2 tmp3 tmp4
    
    %% calculate correlations (15° ori bins)
    corr_base_measured=nan*zeros(size(aa2,1),size(aa2,1));
    corr_bias_measured=nan*zeros(size(aa2,1),size(aa2,1));
    corr_base_dyn=nan*zeros(size(aa2,1),size(aa2,1),4);
    corr_bias_dyn=nan*zeros(size(aa2,1),size(aa2,1),4);
    corr_base_sub=nan*zeros(size(aa2,1),size(aa2,1));
    corr_bias_sub=nan*zeros(size(aa2,1),size(aa2,1));
    sig_base=nan*zeros(size(aa2,1),size(aa2,1));
    sig_bias=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base1=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias1=nan*zeros(size(aa2,1),size(aa2,1));
    noise_base2=nan*zeros(size(aa2,1),size(aa2,1));
    noise_bias2=nan*zeros(size(aa2,1),size(aa2,1));
    z=round(length(oris_u)/2);      % uniform half trial length
    z1=round(z/2);
    z2=round(length(oris_b)/2);
    z3=round(z2/2);
    for i=1:size(aa2,1)-1
        for j=i+1:size(aa2,1)
            % response correlations (signal and noise)
            corr_base_measured(i,j)=akcorrcoef(aa2(i,:)',aa2(j,:)');
            corr_bias_measured(i,j)=akcorrcoef(bb2(i,:)',bb2(j,:)');
            corr_base_dyn(i,j,1)=akcorrcoef(aa2(i,1:z1)',aa2(j,1:z1)');
            corr_base_dyn(i,j,2)=akcorrcoef(aa2(i,z1+1:z)',aa2(j,z1+1:z)');
            corr_base_dyn(i,j,3)=akcorrcoef(aa2(i,z+1:z+z1)',aa2(j,z+1:z+z1)');
            corr_base_dyn(i,j,4)=akcorrcoef(aa2(i,z+z1+1:end)',aa2(j,z+z1+1:end)');
            corr_bias_dyn(i,j,1)=akcorrcoef(bb2(i,1:z3)',bb2(j,1:z3)');
            corr_bias_dyn(i,j,2)=akcorrcoef(bb2(i,z3+1:z2)',bb2(j,z3+1:z2)');
            corr_bias_dyn(i,j,3)=akcorrcoef(bb2(i,z2+1:z2+z3)',bb2(j,z2+1:z2+z3)');
            corr_bias_dyn(i,j,4)=akcorrcoef(bb2(i,z2+z3+1:end)',bb2(j,z2+z3+1:end)');
            corr_base_sub(i,j)=akcorrcoef(ee(i,:)',ee(j,:)');
            corr_bias_sub(i,j)=akcorrcoef(ff(i,:)',ff(j,:)');
            
            % signal correlations: tuning curves
            sig_base(i,j)=akcorrcoef(cc(i,:)',cc(j,:)');
            sig_bias(i,j)=akcorrcoef(dd(i,:)',dd(j,:)');
            
            % noise/spike count correlations
            % alternate way of doing this is z-score responses
            tmp =nan*zeros(1,9);
            tmp2=nan*zeros(1,9);
            tmp3=nan*zeros(1,9);
            tmp4=nan*zeros(1,9);
            tmp5=nan*zeros(1,9);
            tmp6=nan*zeros(1,9);
            for k = 1:9
                [~,sd]=find(stim_resp_u{i,k}>=3);
                [~,sd2]=find(stim_resp_u{j,k}>=3);
                x=setdiff(1:length(stim_resp_u{i,k}),[sd sd2]);
                tmp(k) =akcorrcoef(stim_resp_u{i,k}(x),stim_resp_u{j,k}(x));
                
                [~,sd]=find(stim_resp_b{i,k}>=3);
                [~,sd2]=find(stim_resp_b{j,k}>=3);
                x=setdiff(1:length(stim_resp_b{i,k}),[sd sd2]);
                tmp2(k)=akcorrcoef(stim_resp_b{i,k}(x),stim_resp_b{j,k}(x));
                
                [~,sd]=find(stim_resp_u1{i,k}>=3);
                [~,sd2]=find(stim_resp_u1{j,k}>=3);
                x=setdiff(1:length(stim_resp_u1{i,k}),[sd sd2]);
                tmp3(k)=akcorrcoef(stim_resp_u1{i,k}(x),stim_resp_u1{j,k}(x));
                
                [~,sd]=find(stim_resp_b1{i,k}>=3);
                [~,sd2]=find(stim_resp_b1{j,k}>=3);
                x=setdiff(1:length(stim_resp_b1{i,k}),[sd sd2]);
                tmp4(k)=akcorrcoef(stim_resp_b1{i,k}(x),stim_resp_b1{j,k}(x));
                
                [~,sd]=find(stim_resp_u2{i,k}>=3);
                [~,sd2]=find(stim_resp_u2{j,k}>=3);
                x=setdiff(1:length(stim_resp_u2{i,k}),[sd sd2]);
                tmp5(k)=akcorrcoef(stim_resp_u2{i,k}(x),stim_resp_u2{j,k}(x));
                
                [~,sd]=find(stim_resp_b2{i,k}>=3);
                [~,sd2]=find(stim_resp_b2{j,k}>=3);
                x=setdiff(1:length(stim_resp_b2{i,k}),[sd sd2]);
                tmp6(k)=akcorrcoef(stim_resp_b2{i,k}(x),stim_resp_b2{j,k}(x));
            end
            noise_base(i,j) =mean(tmp);
            noise_bias(i,j) =mean(tmp2);
            noise_base1(i,j)=mean(tmp3);
            noise_bias1(i,j)=mean(tmp4);
            noise_base2(i,j)=mean(tmp5);
            noise_bias2(i,j)=mean(tmp6);
            
            % keep units tuned (>0.3)
            if ob2(i)>0.3
                if ob2(j)>0.3
                    or1=Y2(i);
                    or2=Y2(j);
                    if or1<-75
                        id3=1;
                    elseif or1>=-75 && or1<-60
                        id3=2;
                    elseif or1>=-60 && or1<-45
                        id3=3;
                    elseif or1>=-45 && or1<-30
                        id3=4;
                    elseif or1>=-30 && or1<-15
                        id3=5;
                    elseif or1>=-15 && or1<0
                        id3=6;
                    elseif or1>=0 && or1<15
                        id3=7;
                    elseif or1>=15 && or1<30
                        id3=8;
                    elseif or1>=30 && or1<45
                        id3=9;
                    elseif or1>=45 && or1<60
                        id3=10;
                    elseif or1>=60 && or1<75
                        id3=11;
                    elseif or1>=75
                        id3=12;
                    end
                    if or2<-75
                        id4=1;
                    elseif or2>=-75 && or2<-60
                        id4=2;
                    elseif or2>=-60 && or2<-45
                        id4=3;
                    elseif or2>=-45 && or2<-30
                        id4=4;
                    elseif or2>=-30 && or2<-15
                        id4=5;
                    elseif or2>=-15 && or2<0
                        id4=6;
                    elseif or2>=0 && or2<15
                        id4=7;
                    elseif or2>=15 && or2<30
                        id4=8;
                    elseif or2>=30 && or2<45
                        id4=9;
                    elseif or2>=45 && or2<60
                        id4=10;
                    elseif or2>=60 && or2<75
                        id4=11;
                    elseif or2>=75
                        id4=12;
                    end

                    tmp=find(oris_b==0); % bias ori
                    tmp2=find(oris_b==20); % another ori for comparison
                    temp=tune_b./tune_u;
                    temp(isinf(temp))=nan;
                    if length(tmp)<3*length(tmp2) % if bias <, bias=2:1
                        tmp1_21{id3,id4}=[tmp1_21{id3,id4}; corr_base_measured(i,j)];
                        tmp2_21{id3,id4}=[tmp2_21{id3,id4}; corr_bias_measured(i,j)];
                        tmp1_21_sub{id3,id4}=[tmp1_21_sub{id3,id4}; corr_base_sub(i,j)];
                        tmp2_21_sub{id3,id4}=[tmp2_21_sub{id3,id4}; corr_bias_sub(i,j)];
                        tmp1sig_21{id3,id4}=[tmp1sig_21{id3,id4}; sig_base(i,j)];
                        tmp2sig_21{id3,id4}=[tmp2sig_21{id3,id4}; sig_bias(i,j)];
                        tmp1noise_21{id3,id4}=[tmp1noise_21{id3,id4}; noise_base(i,j)];
                        tmp2noise_21{id3,id4}=[tmp2noise_21{id3,id4}; noise_bias(i,j)];
                        tmp1noise1_21{id3,id4}=[tmp1noise1_21{id3,id4}; noise_base1(i,j)];
                        tmp2noise1_21{id3,id4}=[tmp2noise1_21{id3,id4}; noise_bias1(i,j)];
                        tmp1noise2_21{id3,id4}=[tmp1noise2_21{id3,id4}; noise_base2(i,j)];
                        tmp2noise2_21{id3,id4}=[tmp2noise2_21{id3,id4}; noise_bias2(i,j)];
                        
                    else % if bias >, bias=6:1
                        tmp1_61{id3,id4}=[tmp1_61{id3,id4}; corr_base_measured(i,j)];
                        tmp2_61{id3,id4}=[tmp2_61{id3,id4}; corr_bias_measured(i,j)];
                        tmp1_61_sub{id3,id4}=[tmp1_61_sub{id3,id4}; corr_base_sub(i,j)];
                        tmp2_61_sub{id3,id4}=[tmp2_61_sub{id3,id4}; corr_bias_sub(i,j)];
                        tmp1sig_61{id3,id4}=[tmp1sig_61{id3,id4}; sig_base(i,j)];
                        tmp2sig_61{id3,id4}=[tmp2sig_61{id3,id4}; sig_bias(i,j)];
                        tmp1noise_61{id3,id4}=[tmp1noise_61{id3,id4}; noise_base(i,j)];
                        tmp2noise_61{id3,id4}=[tmp2noise_61{id3,id4}; noise_bias(i,j)];
                        tmp1noise1_61{id3,id4}=[tmp1noise1_61{id3,id4}; noise_base1(i,j)];
                        tmp2noise1_61{id3,id4}=[tmp2noise1_61{id3,id4}; noise_bias1(i,j)];
                        tmp1noise2_61{id3,id4}=[tmp1noise2_61{id3,id4}; noise_base2(i,j)];
                        tmp2noise2_61{id3,id4}=[tmp2noise2_61{id3,id4}; noise_bias2(i,j)];
                        tmp_dyn1{id3,id4}=[tmp_dyn1{id3,id4}; squeeze(corr_base_dyn(i,j,1))];
                        tmp_dyn2{id3,id4}=[tmp_dyn2{id3,id4}; squeeze(corr_base_dyn(i,j,2))];
                        tmp_dyn3{id3,id4}=[tmp_dyn3{id3,id4}; squeeze(corr_base_dyn(i,j,3))];
                        tmp_dyn4{id3,id4}=[tmp_dyn4{id3,id4}; squeeze(corr_base_dyn(i,j,4))];
                        temp_dyn1{id3,id4}=[temp_dyn1{id3,id4}; squeeze(corr_bias_dyn(i,j,1))];
                        temp_dyn2{id3,id4}=[temp_dyn2{id3,id4}; squeeze(corr_bias_dyn(i,j,2))];
                        temp_dyn3{id3,id4}=[temp_dyn3{id3,id4}; squeeze(corr_bias_dyn(i,j,3))];
                        temp_dyn4{id3,id4}=[temp_dyn4{id3,id4}; squeeze(corr_bias_dyn(i,j,4))];
                    end
                end
            end
        end
    end

    %% corr plots for individual files
    aa=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
    aa=aa/sum(aa(:));
   %   plot smoothed measured effect vs prediction from tuning
%     figure
%     supertitle('response correlations (benucci awake)')
%     subplot(231)
%     imagesc(conv2(corr_base_measured,aa,'same'),[-0.3 1])
%     axis square;box off
%     title('Uniform measured')
%     subplot(232)
%     imagesc(conv2(corr_bias_measured,aa,'same'),[-0.3 1])
%     axis square;box off
%     title('Biased measured')
%     subplot(233)
%     imagesc(conv2(corr_bias_measured-corr_base_measured,aa,'same'),[-0.1 0.1])
%     axis square;box off
%     title('Biased-uniform')
%     subplot(234)
%     imagesc(conv2(corr_base_sub,aa,'same'),[-0.3 1])
%     axis square;box off
%     title('Uniform with subsampled bias')
%     subplot(235)
%     imagesc(conv2(corr_bias_sub,aa,'same'),[-0.3 1])
%     axis square;box off
%     title('Biased with subsampled bias')
%     subplot(236)
%     imagesc(conv2(corr_bias_sub-corr_base_sub,aa,'same'),[-0.1 0.1])
%     axis square;box off
%     title('Bias-uniform/bias')
% 
%     figure; plot(Y); 
%     xlabel('sorted channels'); ylabel('ori pref')
% 
%     figure
%     supertitle('signal correlations (benucci awake)')
%     subplot(131)
%     imagesc(conv2(sig_base,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform')
%     subplot(132)
%     imagesc(conv2(sig_bias,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased')
%     subplot(133)
%     imagesc(conv2(sig_bias-sig_base,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
%     
%     figure
%     supertitle('noise correlations (benucci awake)')
%     subplot(131)
%     imagesc(conv2(noise_base,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform')
%     subplot(132)
%     imagesc(conv2(noise_bias,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased')
%     subplot(133)
%     imagesc(conv2(noise_bias-noise_base,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')

    clear tmp tmp1 tmp2 tmp3 tmp4 i j k x y ans cv1 cv2 sd*
    savename=sprintf('%s_corr',filename);
    save(savename)
end
stopbenuccicorrawake

%% plot individual conditional variance histograms
bins=0:1/6:1;
figure
subplot(131)
imagesc(CondVar_u{15,25})
axis square
subplot(132)
imagesc(CondVar_b{15,25})
axis square
subplot(133)
imagesc(CondVar_b{15,25}-CondVar_u{15,25})
axis square

%%

%                 if a<10 % for current set-up, <10 means 2:1 bias experimenet
%                     tmp1_21{id3,id4}=[tmp1_21{id3,id4}; corr_base_measured(i,j)];
%                     tmp2_21{id3,id4}=[tmp2_21{id3,id4}; corr_bias_measured(i,j)];
%                     tmp1_21_sub{id3,id4}=[tmp1_21_sub{id3,id4}; corr_unif_sub(i,j)];
%                     tmp2_21_sub{id3,id4}=[tmp2_21_sub{id3,id4}; corr_bias_sub(i,j)];
%                 else % these are 6:1 bias experiment
%                     tmp1_61{id3,id4}=[tmp1_61{id3,id4}; corr_base_measured(i,j)];
%                     tmp2_61{id3,id4}=[tmp2_61{id3,id4}; corr_bias_measured(i,j)];
%                     tmp1_61_sub{id3,id4}=[tmp1_61_sub{id3,id4}; corr_unif_sub(i,j)];
%                     tmp2_61_sub{id3,id4}=[tmp2_61_sub{id3,id4}; corr_bias_sub(i,j)];
%                 end
%% shuffled analysis - not currently in use

%     %   shuffle trials for each channel/unit
%     for i=1:size(aa2,1)
%         shuf=randperm(length(aa2));
%         shuf2=randperm(length(cc));
%         aas(i,:)=aa2(i,shuf);
%         bbs(i,:)=bb2(i,shuf);
%         ccs(i,:)=cc(i,shuf2);
%         dds(i,:)=dd(i,shuf2);
%     end
%     %   calculate shuffled correlations
%     for i=1:size(aa2,1)-1
%         for j=i+1:size(aa2,1)
%             corr_base_shuf(i,j)=akcorrcoef(aas(i,:),aas(j,:));
%             corr_bias_shuf(i,j)=akcorrcoef(bbs(i,:),bbs(j,:));
%             corr_base_sub_shuf(i,j)=akcorrcoef(ccs(i,:),ccs(j,:));
%             corr_bias_sub_shuf(i,j)=akcorrcoef(dds(i,:),dds(j,:));
%             
%             if ob2(i)>0.11 && ob2(j)>0.11
%                 or1=Y(i);
%                 or2=Y(j);
%                 if or1<-75
%                     id3=1;
%                 elseif or1>=-75 && or1<-60
%                     id3=2;
%                 elseif or1>=-60 && or1<-45
%                     id3=3;
%                 elseif or1>=-45 && or1<-30
%                     id3=4;
%                 elseif or1>=-30 && or1<-15
%                     id3=5;
%                 elseif or1>=-15 && or1<0
%                     id3=6;
%                 elseif or1>=0 && or1<15
%                     id3=7;
%                 elseif or1>=15 && or1<30
%                     id3=8;
%                 elseif or1>=30 && or1<45
%                     id3=9;
%                 elseif or1>=45 && or1<60
%                     id3=10;
%                 elseif or1>=60 && or1<75
%                     id3=11;
%                 elseif or1>=75
%                     id3=12;
%                 end
%                 if or2<-75
%                     id4=1;
%                 elseif or2>=-75 && or2<-60
%                     id4=2;
%                 elseif or2>=-60 && or2<-45
%                     id4=3;
%                 elseif or2>=-45 && or2<-30
%                     id4=4;
%                 elseif or2>=-30 && or2<-15
%                     id4=5;
%                 elseif or2>=-15 && or2<0
%                     id4=6;
%                 elseif or2>=0 && or2<15
%                     id4=7;
%                 elseif or2>=15 && or2<30
%                     id4=8;
%                 elseif or2>=30 && or2<45
%                     id4=9;
%                 elseif or2>=45 && or2<60
%                     id4=10;
%                 elseif or2>=60 && or2<75
%                     id4=11;
%                 elseif or2>=75
%                     id4=12;
%                 end
%                 % for previous set-up w/ missing 140°, <10 means 2:1 bias experimenet
% %                 if a<10 
% %                     tmp1s_21_shuf{id3,id4}=[tmp1s_21_shuf{id3,id4}; corr_base_shuf(i,j)];
% %                     tmp2s_21_shuf{id3,id4}=[tmp2s_21_shuf{id3,id4}; corr_bias_shuf(i,j)];
% %                     tmp1s_21_sub_shuf{id3,id4}=[tmp1s_21_sub_shuf{id3,id4}; corr_unif_sub_shuf(i,j)];
% %                     tmp2s_21_sub_shuf{id3,id4}=[tmp2s_21_sub_shuf{id3,id4}; corr_bias_sub_shuf(i,j)];
% %                 else % these are 6:1 bias experiment
% %                     tmp1s_61_shuf{id3,id4}=[tmp1s_61_shuf{id3,id4}; corr_base_shuf(i,j)];
% %                     tmp2s_61_shuf{id3,id4}=[tmp2s_61_shuf{id3,id4}; corr_bias_shuf(i,j)];
% %                     tmp1s_61_sub_shuf{id3,id4}=[tmp1s_61_sub_shuf{id3,id4}; corr_unif_sub_shuf(i,j)];
% %                     tmp2s_61_sub_shuf{id3,id4}=[tmp2s_61_sub_shuf{id3,id4}; corr_bias_sub_shuf(i,j)];
% %                 end
%                 % for current set-up:
%                 tmp=find(oris_b==0); % bias ori
%                 tmp2=find(oris_b==20); % another ori for comparison
%                 if length(tmp)<3.5*length(tmp2) % if bias <, bias=2:1
%                     tmp1s_21{id3,id4}=[tmp1s_21{id3,id4}; corr_base_shuf(i,j)];
%                     tmp2s_21{id3,id4}=[tmp2s_21{id3,id4}; corr_bias_shuf(i,j)];
%                     tmp1s_21_sub{id3,id4}=[tmp1s_21_sub{id3,id4}; corr_base_sub_shuf(i,j)];
%                     tmp2s_21_sub{id3,id4}=[tmp2s_21_sub{id3,id4}; corr_bias_sub_shuf(i,j)];
%                 else % if bias >, bias=6:1
%                     tmp1s_61{id3,id4}=[tmp1s_61{id3,id4}; corr_base_shuf(i,j)];
%                     tmp2s_61{id3,id4}=[tmp2s_61{id3,id4}; corr_bias_shuf(i,j)];
%                     tmp1s_61_sub{id3,id4}=[tmp1s_61_sub{id3,id4}; corr_base_sub_shuf(i,j)];
%                     tmp2s_61_sub{id3,id4}=[tmp2s_61_sub{id3,id4}; corr_bias_sub_shuf(i,j)];
%                 end
%             end
%         end
%     end
%     %   calculate difference between measured and shuffled correlations
%     r_u_diff=corr_base_measured-corr_base_shuf;
%     r_b_diff=corr_bias_measured-corr_bias_shuf;
%     r_u_sub_diff=corr_base_sub-corr_base_sub_shuf;
%     r_b_sub_diff=corr_bias_sub-corr_bias_sub_shuf;
%     %   plot smoothed measured effect vs prediction from tuning(shuffled)
%     figure
%     supertitle('shuffled correlations (benucci awake)')
%     subplot(231)
%     imagesc(conv2(corr_base_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform measured')
%     subplot(232)
%     imagesc(conv2(corr_bias_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased measured')
%     subplot(233)
%     imagesc(conv2(corr_bias_shuf-corr_base_shuf,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
%     subplot(234)
%     imagesc(conv2(corr_base_sub_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform with subsampled bias')
%     subplot(235)
%     imagesc(conv2(corr_bias_sub_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased with subsampled bias')
%     subplot(236)
%     imagesc(conv2(corr_bias_sub_shuf-corr_base_sub_shuf,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Bias-uniform/bias')
%     
%     %   plot difference between measured and shuffled correlations
%     figure
%     supertitle('diff(measured-shuffled) correlations (benucci awake)')
%     subplot(231)
%     imagesc(conv2(r_u_diff,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform measured')
%     subplot(232)
%     imagesc(conv2(r_b_diff,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased measured')
%     subplot(233)
%     imagesc(conv2(r_b_diff-r_u_diff,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Biased-uniform')
%     subplot(234)
%     imagesc(conv2(r_u_sub_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Uniform with subsampled bias')
%     subplot(235)
%     imagesc(conv2(r_b_sub_shuf,aa,'same'),[-0.2 0.75])
%     axis square;box off
%     title('Biased with subsampled bias')
%     subplot(236)
%     imagesc(conv2(r_b_sub_shuf-r_u_sub_shuf,aa,'same'),[-0.05 0.075])
%     axis square;box off
%     title('Bias-uniform/bias')