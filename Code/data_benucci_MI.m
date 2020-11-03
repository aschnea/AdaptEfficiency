%%  do mutual information for each channel/unit
clear
% DO WITH AND WITHOUT _DROP
% % % % 2:1
% load('129r001p173_preprocess')    %   2:1 with 160 over-represented
% load('130l001p169_preprocess')    %   2:1
% load('140l001p107_preprocess')
% load('140l001p122_preprocess')
% load('140r001p105_preprocess')
% load('140r001p122_preprocess')
% % % % 4:1
% load('130l001p170_preprocess')    %   4:1, 0 is overrepresented
% load('140l001p108_preprocess')
% load('140l001p110_preprocess')    % bias shift
% load('140r001p107_preprocess')
% load('140r001p109_preprocess')    % bias shift
% % % % low contrast
% load('lowcon114_preprocess')
% load('lowcon115_preprocess')
% load('lowcon116_preprocess')
% load('lowcon117_preprocess')
% % % % awake time
% load('140l113_awaketime_preprocess')
% load('140r113_awaketime_preprocess')
% % % % blank
% load('140l111_blank_preprocess')
% load('140r111_blank_preprocess')
% load('140r161_blank_preprocess')
% % % % benucci
% load('140l118_benucci_preprocess')
% load('140r118_benucci_preprocess')
% CURRENTLY ONLY SINGLE UNIT MI CALCULATED

for n=1:33
    clearvars -except n
    if n==1
        load('129r001p173_preprocess')
        name='129r001p173_entropy';
    elseif n==2
        load('130l001p169_preprocess')
        name='130l001p169_entropy';
    elseif n==3
        load('140l001p107_preprocess')
        name='140l001p107_entropy';
    elseif n==4
        load('140l001p122_preprocess')
        name='140l001p122_entropy';
    elseif n==5
        load('140r001p105_preprocess')
        name='140r001p105_entropy';
    elseif n==6
        load('140r001p122_preprocess')
        name='140r001p122_entropy';
    elseif n==7
        load('130l001p170_preprocess')
        name='130l001p170_entropy';
    elseif n==8
        load('140l001p108_preprocess')
        name='140l001p108_entropy';
    elseif n==9
        load('140l001p110_preprocess')
        name='140l001p110_entropy';
    elseif n==10
        load('140r001p107_preprocess')
        name='140r001p107_entropy';
    elseif n==11
        load('140r001p109_preprocess')
        name='140r001p109_entropy';
    elseif n==12
        load('lowcon114_preprocess')
        name='lowcon114_entropy';
    elseif n==13
        load('lowcon115_preprocess')
        name='lowcon115_entropy';
    elseif n==14
        load('lowcon116_preprocess')
        name='lowcon116_entropy';
    elseif n==15
        load('lowcon117_preprocess')
        name='lowcon117_entropy';
    elseif n==16
        load('140l113_awaketime_preprocess')
        name='140l113_awaketime_entropy';
    elseif n==17
        load('140r113_awaketime_preprocess')
        name='140r113_awaketime_entropy';
        
    elseif n==18 % start of experiment 141 files
        load('141r001p006_awaketime_preprocess')
        name='141r001p006_awaketime_entropy';
    elseif n==19
        load('141r001p007_awaketime6_preprocess')
        name='141r001p007_awaketime6_entropy';
    elseif n==20
        load('141r001p009_awaketime_fine_preprocess')
        name='141r001p009_awaketime_fine_entropy';
    elseif n==21 % rotated AT 4:1 (80°)
        load('141r001p024_awaketime_preprocess')
        name='141r001p024_awaketime_entropy';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_preprocess')
        name='141r001p025_awaketime6_entropy';
    elseif n==23 % rotated AT fineori (90°??)
        load('141r001p027_awaketime_fine_preprocess')
        name='141r001p027_awaketime_fine_entropy';
    elseif n==24 % rotated fineori (40°)
        load('141r001p038_awaketime_fine_preprocess')
        name='141r001p038_awaketime_fine_entropy';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_preprocess')
        name='141r001p039_awaketime6_entropy';
    elseif n==26 % rotated awaketime 4:1 (120°)
        load('141r001p041_awaketime_preprocess')
        name='141r001p041_awaketime_entropy';
    elseif n==27
        load('141r001p114_preprocess')
        name='141r001p114_entropy';
        
    elseif n==28    % start of 142 files
        load('142l001p002_awaketime_preprocess')
        name='142l001p002_awaketime_entropy';
    elseif n==29
        load('142l001p004_awaketime_fine_preprocess')
        name='142l001p004_awaketime_fine_entropy';
    elseif n==30
        load('142l001p006_awaketime6_preprocess')
        name='142l001p006_awaketime6_entropy';
    elseif n==31
        load('142l001p007_awaketime_preprocess')
        name='142l001p007_awaketime_entropy';
    elseif n==32
        load('142l001p009_awaketime_fine_preprocess')
        name='142l001p009_awaketime_fine_entropy';
    elseif n==33
        load('142l001p010_awaketime6_preprocess')
        name='142l001p010_awaketime6_entropy';
    end
%     clearvars -except spikes a stim uniq_stim syncs n name
    %% look at ori tuning and preference
%     ori_base=[(0:20:160) 200]; % already included in *_preprocess
    % resp_raw_base; resp_raw_bias
    for i=1:size(resp_raw_base,1)
        for j=1:length(ori_base)
            temp=find(base==ori_base(j));
            resp_base(i,j)=mean(resp_raw_base(i,temp)); %tuning curves
            if j==length(ori_base)
                if isempty(temp)    % empty for blank and benucci files
                    resp_base(i,j)=spont_fr(i);
                    spont_sem_u(i)=std(spont_tmp{i}(:))/sqrt(length(spont_tmp{i}));
                else
                    spont_sem_u(i)=std(resp_raw_base(i,temp))/sqrt(length(temp));
                end
            end
            temp=find(bias==ori_base(j));
            resp_bias(i,j)=mean(resp_raw_bias(i,temp));
            if j==length(ori_base)
                if isempty(temp)    % empty for blank and benucci files
                    resp_bias(i,j)=spont_fr(i);
                    spont_sem_b(i)=std(spont_tmp{i}(:))/sqrt(length(spont_tmp{i}));
                else
                    spont_sem_b(i)=std(resp_raw_bias(i,temp))/sqrt(length(temp));
                end
            end
        end
    end
    
    % group neurons by their ori pref
    for i=1:size(resp_base,1)
        [~,~,oribias(i),oripref(i),~,~] = orivecfit(ori_base(1:end-1),resp_base(i,1:end-1),resp_base(i,end));
        [~,~,oribiasa(i),oriprefa(i),~,~]=orivecfit(ori_base(1:end-1),resp_bias(i,1:end-1),resp_bias(i,end));
        % for benucci and blank version use:
        %     [~,~,oribias(i),oripref(i),~,~] = orivecfit((0:20:160),resp_base(i,1:9),spont_fr(i));
        %     [~,~,oribiasa(i),oriprefa(i),~,~] = orivecfit((0:20:160),resp_bias(i,1:9),spont_fr(i));
    end
%     oripref(oripref>180)=oripref(oripref>180)-180;
%     oriprefa(oriprefa>180)=oriprefa(oriprefa>180)-180;
    oripref(oripref<0)=oripref(oripref<0)+180;
    oriprefa(oriprefa<0)=oriprefa(oriprefa<0)+180;  %all oriprefs = 0:180
    
    [Y,I]=sort(oripref);
    val_mode=mode(bias); %should be 0 or 80
%     x=val_mode+90;
    ori_realign=abs(Y-val_mode);
    ori_realign(ori_realign<0)=ori_realign(ori_realign<0)+180;
    
    
    %% identify responsive units
    for e=1:size(resp_base,1)
        if max(resp_base(e,1:end-1))>(resp_base(e,end)+3*spont_sem_u(e))
            responsive(e)=1;
        else
            responsive(e)=0;
        end
    end
    responsive=logical(responsive);
    resp_base=resp_base(responsive,:);
    resp_bias=resp_bias(responsive,:);
    resp_raw_bias=resp_raw_bias(responsive,:);
    resp_raw_base=resp_raw_base(responsive,:);
    oripref=oripref(responsive);
    oribias=oribias(responsive);
    spont_sem_u=spont_sem_u(responsive);
    spont_sem_b=spont_sem_b(responsive);
    
    %% calculate entropy
    for i=1:size(resp_raw_base,1)
        %     if responsive(i)==1
        clear tmp* bins a* entr* cond*
        % store all responses for unit i
        tmp=resp_raw_base(i,:);             % uniform response vector
        tmp2=resp_raw_bias(i,:);            % bias response vector
        % find response range for unit it
        bins=(min([tmp(:);tmp2(:)]):1:max([tmp(:);tmp2(:)]));   % integer spike counts
        % create response probability histogram
        a=histc(tmp,bins)/length(tmp);      %P(r) uniform
        a2=histc(tmp2,bins)/length(tmp2);   %P(r) bias
        
        %   calculate response entropy of unit i
        for j=1:length(bins)
            entr_base(j)=a(j)*log2(a(j));
            entr_bias(j)=a2(j)*log2(a2(j));
        end
        
        %   calculate conditional entropy (Resp|stimulus)
        for j=1:length(ori_base)-1              % oris used, excludes blanks
            xx=find(base==ori_base(j));         % cases of each ori
            a_xx=histc(tmp(xx),bins)/length(xx);% P(r) in each bin for given ori
            p_s_base1(j)=length(xx)/length(base);% ratio of given ori to total trials
            for k=1:length(bins)
                cond_ent_base(j,k)=a_xx(k)*log2(a_xx(k));   % conditional entropy for uniform
            end
            
            xx=find(bias==ori_base(j));
            a2_xx=histc(tmp2(xx),bins);
            a2_xx=a2_xx/sum(a2_xx);
            p_s_bias1(j)=length(xx)/length(bias);
            for k=1:length(bins)
                cond_ent_bias(j,k)=a2_xx(k)*log2(a2_xx(k)); % conditional entropy for bias
            end
        end
        
        % Store Entropy and Conditional entropy in neuron (pref) domain
        H_base(i)=-1*nansum(entr_base); % measured entropy
        CondH_base(i)=sum(p_s_base1.*nansum(cond_ent_base,2)');  % predicted entropy
        H_bias(i)=-1*nansum(entr_bias);
        CondH_bias(i)=sum(p_s_bias1.*nansum(cond_ent_bias,2)');
        aa(i)=H_bias(i)+CondH_bias(i);   % MI bias (neuron-domain)
        bb(i)=H_base(i)+CondH_base(i);   % MI base
        
        % Store components of conditional entropy for MI-stimulus domain
%         CondH_base_stim(i,:)=(p_s_base1.*nansum(cond_ent_base,2)');  % entropy per stim
%         CondH_bias_stim(i,:)=(p_s_bias1.*nansum(cond_ent_bias,2)');  % entropy per stim
        CondH_base_stim(i,:)=(nansum(cond_ent_base,2)');  % entropy per stim
        CondH_bias_stim(i,:)=(nansum(cond_ent_bias,2)');  % entropy per stim
        aa_stim(i,:)=H_bias(i)+CondH_bias_stim(i,:);  % MI bias (stim domain)
        bb_stim(i,:)=H_base(i)+CondH_base_stim(i,:);  % MI base (stim domain)
        %     end
    end
    
    %%  make predictions for unadapted responses
    % subsampled trials to create matched distributions (used in
    % correlations & LDA analysis)
    
    %   find the biased ori
    a=histc(bias,ori_base);
    over_ori=ori_base(find(a==max(a)));
    over_ori_ind=find(a==max(a));
    over_rat=max(a)/min(a);
%     
%     %   make a biased response distribution in the base condition
%     keep=find(base==over_ori);
%     keep=keep(:);
%     keep2=find(bias==over_ori);
%     keep2=keep2(randperm(length(keep2),length(keep)))';
%     target=round(length(keep)/over_rat);
%     for i=1:length(ori_base)-1
%         if i~=over_ori_ind
%             tmp=find(base==ori_base(i));
%             keep=[keep;tmp(randperm(target))'];
%             % this is the subset of trials to keep in uniform to get the same stimulus distribution as biased
%             temp=find(bias==ori_base(i));
%             keep2=[keep2;temp(randperm(target))'];
%         end
%     end
    
    %   make a uniform response distribution in both environments:
    keep=[];
    keep2=[];
    min_count=min(a);   % Trials of each ori to create distribution
    for i=1:length(ori_base)-1
        % trials to keep from uniform
        tmp=find(base==ori_base(i));
        keep=[keep;tmp(randperm(min_count))'];
        % trials to keep from bias
        temp=find(bias==ori_base(i));
        keep2=[keep2;temp(randperm(min_count))'];
    end
    
    clear tmp* bins entr* cond*
    %%   compute MI for these uniform trials (matched distributions)
    for i=1:size(resp_raw_base,1)
        tmp=resp_raw_base(i,keep);  % kept responses
        tmp2=resp_raw_bias(i,keep2);
        base_pred=base(keep);       % kept stimuli
        bias_pred=bias(keep2);
        bins=(min([tmp(:);tmp2(:)]):1:max([tmp(:);tmp2(:)])); % Response range
        a=histc(tmp,bins)/length(tmp);      %P(r)
        a2=histc(tmp2,bins)/length(tmp2);
        % response entropy: 
        for j=1:length(bins)                
            entr_base(j)=a(j)*log2(a(j));
            entr_bias(j)=a2(j)*log2(a2(j));
        end
        % conditional entropy:
        for j=1:length(ori_base)-1
            xx=find(base_pred==ori_base(j)); %find index of each ori
            a_xx=histc(tmp(xx),bins)/length(xx); %P(r) for each ori
            p_s_base2(j)=length(xx)/length(base_pred); %P(ori)
            for k=1:length(bins)
                cond_ent_base(j,k)=a_xx(k)*log2(a_xx(k));
            end
            
            xx=find(bias_pred==ori_base(j));
            a2_xx=histc(tmp2(xx),bins);
            a2_xx=a2_xx/sum(a2_xx);
            p_s_bias2(j)=length(xx)/length(bias_pred);
            for k=1:length(bins)
                cond_ent_bias(j,k)=a2_xx(k)*log2(a2_xx(k));
            end
        end
        % Store Entropy and Conditional entropy in neuron (pref) domain
        H_base_pred(i)=-1*nansum(entr_base); % measured entropy
        CondH_base_pred(i)=sum(p_s_base2.*nansum(cond_ent_base,2)');% predicted entropy
        H_bias_pred(i)=-1*nansum(entr_bias);
        CondH_bias_pred(i)=sum(p_s_bias2.*nansum(cond_ent_bias,2)');
        aap(i)=H_bias_pred(i)+CondH_bias_pred(i); % MI bias (neuron-domain)
        bbp(i)=H_base_pred(i)+CondH_base_pred(i); % MI base
        
        % Store components of conditional entropy for MI-stimulus domain
%         CondH_base_stim_pred(i,:)=(p_s_base2.*nansum(cond_ent_base,2)');  % entropy per stim
%         CondH_bias_stim_pred(i,:)=(p_s_bias2.*nansum(cond_ent_bias,2)');  % entropy per stim
        CondH_base_stim_pred(i,:)=(nansum(cond_ent_base,2)');  % entropy per stim
        CondH_bias_stim_pred(i,:)=(nansum(cond_ent_bias,2)');  % entropy per stim
        aap_stim(i,:)=H_bias_pred(i)+CondH_bias_stim_pred(i,:);  % MI bias (stim domain)
        bbp_stim(i,:)=H_base_pred(i)+CondH_base_stim_pred(i,:);  % MI base (stim domain)
        
        
        % correct for finite data bias in entropy
        % Hexp=Htrue+sum_a(c_a/N^a) - c is weighting coefficient for a and
        % N is # of times each stimulus was repeated
        options = optimset('TolFun',1e-5,'TolX',1e-4,'Maxiter',10000,...
            'MaxFunEvals',10000,'Display','off','LargeScale','off');
        N=min_count;
        % Response Entropy:
        startu=[H_base_pred(i) 1 1 1];
        [H_u_corrected(i,:), error_u(i)] = fminsearch(@(b) Hcorrection(b, H_base_pred(i), N),startu,options);
        startb=[H_bias_pred(i) 1 1 1];
        [H_b_corrected(i,:), error_b(i)] = fminsearch(@(b) Hcorrection(b, H_bias_pred(i), N),startb,options);
        % conditional entropy (neuron domain):
        startu=[CondH_base_pred(i) 1 1 1];
        [CondH_u_corrected(i,:), error_Condu(i)] = fminsearch(@(b) Hcorrection(b, CondH_base_pred(i), N),startu,options);
        startb=[CondH_bias_pred(i) 1 1 1];
        [CondH_b_corrected(i,:), error_Condb(i)] = fminsearch(@(b) Hcorrection(b, CondH_bias_pred(i), N),startb,options);
        % conditional entropy (stimulus domain):
        for j = 1:length(ori_base)-1
            startu=[CondH_base_stim_pred(i,j) 1 1 1];
            [CondH_u_corrected_stim(i,j,:), error_Condu(i,j)] = fminsearch(@(b) Hcorrection(b, CondH_base_stim_pred(i,j), N),startu,options);
            startb=[CondH_bias_stim_pred(i,j) 1 1 1];
            [CondH_b_corrected_stim(i,j,:), error_Condb(i,j)] = fminsearch(@(b) Hcorrection(b, CondH_bias_stim_pred(i,j), N),startb,options);
        end
        
        
        aap_crct(i)=H_b_corrected(i)+CondH_b_corrected(i);
        bbp_crct(i)=H_u_corrected(i)+CondH_u_corrected(i);
        aap_stim_crct(i,:)=H_b_corrected(i)+squeeze(CondH_b_corrected_stim(i,:,1));
        bbp_stim_crct(i,:)=H_u_corrected(i)+squeeze(CondH_u_corrected_stim(i,:,1));
        clear tmp* bins entr* cond* start*
    end
%     figure; subplot(221); plot(aap_crct,aap,'.'); refline(1,0);
%     subplot(222); plot(bbp_crct,bbp,'.'); refline(1,0);
%     subplot(223); plot(bbp,aap,'.'); refline(1,0);
%     subplot(224);plot(bbp_crct,aap_crct,'.');refline(1,0)
    %%  entropy plots
    
%     figure
%     subplot(321)
%     plot(H_bias+CondH_bias,H_base+CondH_base,'.k')
%     axis([0 1 0 1])
%     axis square;box off
%     hold on
%     plot((0:0.1:1),(0:0.1:1),':r')
%     title('measured')
%     xlabel('MI bias')
%     ylabel('MI base')
%     
%     subplot(323)
%     tmp=oripref; tmp(tmp<90)=tmp(tmp<90)+180;
%     plot(tmp,aa-bb,'.k')
%     hold on;box off
%     plot((90:270),zeros(181,1),':r')
%     xlabel('Oripref (deg)')
%     ylabel('MI_b_i_a_s-MI_b_a_s_e')
%     set(gca,'XTick',[90 135 180 225 270],'XTickLabel',{'-90','-45','0','45','90'})
%     
%     subplot(322)
%     plot(H_bias_pred+CondH_bias_pred,H_base_pred+CondH_base_pred,'.k')
%     axis([0 4 0 4])
%     axis square;box off
%     hold on
%     plot((0:0.1:4),(0:0.1:4),':r')
%     title('Predicted: UB vs AB')
%     xlabel('MI bias')
%     
%     subplot(324)
%     plot(tmp,aap-bbp,'.k')
%     hold on;box off
%     plot((90:270),zeros(181,1),':r')
%     set(gca,'XTick',[90 135 180 225 270],'XTickLabel',{'-90','-45','0','45','90'})
%     xlabel('Oripref (deg)')
%     supertitle('single')
%     
%     %   poster plots
%     subplot(325)
%     frac=100*(aap-bbp)./bbp;
%     bins=(-20:2.5:20);
%     a=histc(frac,bins);
%     bar(bins+diff(bins(1:2))/2,a/sum(a),1);
%     hold on;box off
%     plot(mean(frac),0.4,'vk')
%     set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-22.5 22.5])
%     xlabel('Predicted: Change in MI (%)')
%     ylabel('Proportion of cases')
%     
%     subplot(326)
%     % oripref_bins=[168.75 11.25:22.5:168.75];
%     oripref_bins=[78.75 -78.75:22.5:78.75];
%     for i=1:length(oripref_bins)-1
%         if i==1
%             tmp=frac(find(oripref>oripref_bins(i)));
%             tmp2=frac(find(oripref<oripref_bins(i+1)));
%             tmp=union(tmp,tmp2);
%         else
%             tmp=frac(find(oripref>oripref_bins(i) & oripref<oripref_bins(i+1)));
%         end
%         dmi(i)=mean(tmp);
%         dmi_se(i)=std(tmp)/length(tmp);
%     end
%     plotbins=(-90:22.5:90);
%     errorline(plotbins(1:end-1),circshift(dmi,[0 8]),circshift(dmi_se,[0 8]),'k')
%     hold on;box off
%     plot((-90:90),zeros(181,1),':k')
%     xlabel('Oripref (deg)')
%     set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
%     ylabel('Predicted: % change in MI')
%     xlabel('Orientation preference (deg)')
%     supertitle('single')
%     stop
    clear e ans dmi* frac i j k over* sort_codes spikes target temp tmp* xx y stim syncs uniq_stim resp_base_test resp_bias_test bins
    save(name);
end
stop

%% shuffle responses for I_shuffle
% for i = 1:size(resp_raw_base,1)
%     shuf=randperm(size(resp_raw_base,2));
%     shuf2=randperm(size(resp_raw_bias,2));
%     shuf_u(i,:)=resp_raw_base(i,shuf);
%     shuf_b(i,:)=resp_raw_bias(i,:);
%     ori_u_s(i,:)=base(shuf);  % keep each list of orientations for CondH
%     ori_b_s(i,:)=bias(shuf2);
% end
stop

%% pairwise MI/redundancy
clear
for n=[19 22 25 30 33]%18:27
    clearvars -except n
    if n==1
        load('129r001p173_entropy')
        name='129r001p173_jentropy';
    elseif n==2
        load('130l001p169_entropy')
        name='130l001p169_jentropy';
    elseif n==3
        load('140l001p107_entropy')
        name='140l001p107_jentropy';
    elseif n==4
        load('140l001p122_entropy')
        name='140l001p122_jentropy';
    elseif n==5
        load('140r001p105_entropy')
        name='140r001p105_jentropy';
    elseif n==6
        load('140r001p122_entropy')
        name='140r001p122_jentropy';
    elseif n==7
        load('130l001p170_entropy')
        name='130l001p170_jentropy';
    elseif n==8
        load('140l001p108_entropy')
        name='140l001p108_jentropy';
    elseif n==9
        load('140l001p110_entropy')
        name='140l001p110_jentropy';
    elseif n==10
        load('140r001p107_entropy')
        name='140r001p107_jentropy';
    elseif n==11
        load('140r001p109_entropy')
        name='140r001p109_jentropy';
    elseif n==12
        load('lowcon114_entropy')
        name='lowcon114_jentropy';
    elseif n==13
        load('lowcon115_entropy')
        name='lowcon115_jentropy';
    elseif n==14
        load('lowcon116_entropy')
        name='lowcon116_jentropy';
    elseif n==15
        load('lowcon117_entropy')
        name='lowcon117_jentropy';
    elseif n==16
        load('140l113_awaketime_entropy')
        name='140l113_awaketime_jentropy';
    elseif n==17
        load('140r113_awaketime_entropy')
        name='140r113_awaketime_jentropy';
        
    elseif n==18 % start of experiment 141 files
        load('141r001p006_awaketime_entropy')
        name='141r001p006_awaketime_jentropy';
    elseif n==19
        load('141r001p007_awaketime6_entropy')
        name='141r001p007_awaketime6_jentropy';
    elseif n==20
        load('141r001p009_awaketime_fine_entropy')
        name='141r001p009_awaketime_fine_jentropy';
    elseif n==21 % rotated AT 4:1 (80°)
        load('141r001p024_awaketime_entropy')
        name='141r001p024_awaketime_jentropy';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_entropy')
        name='141r001p025_awaketime6_jentropy';
    elseif n==23 % rotated AT fineori (90°??)
        load('141r001p027_awaketime_fine_entropy')
        name='141r001p027_awaketime_fine_jentropy';
    elseif n==24 % rotated fineori (40°)
        load('141r001p038_awaketime_fine_entropy')
        name='141r001p038_awaketime_fine_jentropy';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_entropy')
        name='141r001p039_awaketime6_jentropy';
    elseif n==26 % rotated awaketime 4:1 (120°)
        load('141r001p041_awaketime_entropy')
        name='141r001p041_awaketime_jentropy';
    elseif n==27
        load('141r001p114_entropy')
        name='141r001p114_jentropy';
     elseif n==28    % start of 142 files
        load('142l001p002_awaketime_entropy')
        name='142l001p002_awaketime_jentropy';
    elseif n==29
        load('142l001p004_awaketime_fine_entropy')
        name='142l001p004_awaketime_fine_jentropy';
    elseif n==30
        load('142l001p006_awaketime6_entropy')
        name='142l001p006_awaketime6_jentropy';
    elseif n==31
        load('142l001p007_awaketime_entropy')
        name='142l001p007_awaketime_jentropy';
    elseif n==32
        load('142l001p009_awaketime_fine_entropy')
        name='142l001p009_awaketime_fine_jentropy';
    elseif n==33
        load('142l001p010_awaketime6_entropy')
        name='142l001p010_awaketime6_jentropy';
    end
    
    
    
    % sort oris and matrices for work below
    val_mode=mode(bias);
    oripref(oripref<0)=oripref(oripref<0)+180;
    oriprefa(oriprefa<0)=oriprefa(oriprefa<0)+180;  %all oriprefs = 0:180
    
    % re-sort units by oripref; reorganize so oripref of 0 equals adapter
    ori_realign=oripref-val_mode;
    ori_realign(ori_realign<0)=ori_realign(ori_realign<0)+180; % make range 0:180 again
    [Y,I]=sort(ori_realign);
    
    tmp=find(ori_realign==min(ori_realign));
    Y2=[Y(tmp(1):end) Y(1:tmp(1))];
    I2=[I(tmp(1):end) I(1:tmp(1)-1)];
    resp_base_sort=resp_raw_base(I2,:);
    resp_bias_sort=resp_raw_bias(I2,:);
    % shuf_u_sort=shuf_u(I2,:);
    % shuf_b_sort=shuf_b(I2,:);
    aa_sort=aa(I2);
    bb_sort=bb(I2);
    aap_sort=aap_crct(I2);
    bbp_sort=bbp_crct(I2);
%     H_base_sort=H_base(I2);
%     H_bias_sort=H_bias(I2);
%     H_basep_sort=H_base_pred(I2);
%     H_biasp_sort=H_bias_pred(I2);
    
    % work for information:
    % I(X,Y)=KL(P(x,y)|P(x)*P*y)) = I(X,Y)=sum_xy(P(x,y)*log2(P(x,y)/P(x)P(y)))
    % I(X,Y)=H(X)+H(Y)-H(X,Y)
    % measure redundancy as I_joint-(I_1+I_2).
    uMI=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
    bMI=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     uMI2=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     bMI2=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     uMI3=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     bMI3=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
    uMI_pred=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
    bMI_pred=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     uMI_pred2=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     bMI_pred2=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     uMI_pred3=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
%     bMI_pred3=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
    % % % % shuffled Information only useful for single class/ori. Ignoring it
    % uMI_shuf=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));
    % bMI_shuf=nan*zeros(size(resp_raw_base,1),size(resp_raw_base,1));

    for i=1:size(resp_base_sort,1)-1
        clear tmp* temp* bin* u u1 u2 b b1 b2 u_s b_s p_* cond_* resp1 resp2 resp3 resp4...
            us1 bs1 us2 bs2 u_shuf b_shuf

        tmp=resp_base_sort(i,keep);
        tmp2=resp_bias_sort(i,keep2);
        resp1=resp_base_sort(i,:);
        resp2=resp_bias_sort(i,:);
    %     us1=shuf_u_sort(i,:);
    %     bs1=shuf_b_sort(i,:);
        bins1=(0:1:max([resp1(:);resp2(:)])); % range of integer spike counts

        if length(bins1)>length(tmp)/9
            disp('Should bin')
        end   

        for j = i+1:size(resp_base_sort,1)
            temp=resp_base_sort(j,keep);
            temp2=resp_bias_sort(j,keep2);
            resp3=resp_base_sort(j,:);
            resp4=resp_bias_sort(j,:);
    %         us2=shuf_u_sort(j,:);
    %         bs2=shuf_b_sort(j,:);
            bins2=(0:1:max([resp3(:);resp4(:)]));% range of integer spike counts

            % 2D joint probability matrix: P(x,y)
            u_pred=histcounts2(tmp,temp,bins1,bins2,'Normalization','Probability');  %P(r) uniform
            b_pred=histcounts2(tmp2,temp2,bins1,bins2,'Normalization','Probability'); %P(r) bias
            u=histcounts2(resp1,resp3,bins1,bins2,'Normalization','Probability');  %P(r) uniform
            b=histcounts2(resp2,resp4,bins1,bins2,'Normalization','Probability'); %P(r) bias
    %         u_shuf=histcounts2(us1,us2,bins1,bins2,'Normalization','Probability');
    %         b_shuf=histcounts2(bs1,bs2,bins1,bins2,'Normalization','Probability');

            % calculation joint entropy (H):
            H_u(i,j)=-1*nansum(nansum(u.*log2(u)));
            H_b(i,j)=-1*nansum(nansum(b.*log2(b)));
            H_u_pred(i,j)=-1*nansum(nansum(u_pred.*log2(u_pred)));
            H_b_pred(i,j)=-1*nansum(nansum(b_pred.*log2(b_pred)));
    %         H_u_shuf(i,j)=-1*nansum(nansum(u_shuf.*log2(u_shuf)));
    %         H_b_shuf(i,j)=-1*nansum(nansum(b_shuf.*log2(b_shuf)));

            % calculate conditional joint entropy (condH):
            for k=1:length(ori_base)-1
                xx=find(base==ori_base(k)); % # of trials of each ori
                p_s_base(k)=length(xx)/length(base); % ratio of given ori to total trials
                % joint probability matrix for stimulus k for units i, j:
                u_s=histcounts2(resp1(xx),resp3(xx),bins1,bins2,'Normalization','Probability');
                cond_ent_base(k)=nansum(nansum(u_s.*log2(u_s)));

                xx=find(bias==ori_base(k));
                p_s_bias(k)=length(xx)/length(bias);
                b_s=histcounts2(resp2(xx),resp4(xx),bins1,bins2,'Normalization','Probability');
                cond_ent_bias(k)=nansum(nansum(b_s.*log2(b_s)));
            end
            condH_u(i,j)=nansum(p_s_base.*cond_ent_base);
            condH_b(i,j)=nansum(p_s_bias.*cond_ent_bias);

            % repeat conditional entropy for prediction (subsample)
            for k=1:length(ori_base)-1
                xx=find(base_pred==ori_base(k)); % # of trials of each ori
                p_s_base(k)=length(xx)/length(base_pred); % ratio of given ori to total trials
                u_s=histcounts2(tmp(xx),temp(xx),bins1,bins2,'Normalization','Probability');
                cond_ent_base_pred(k)=nansum(nansum(u_s.*log2(u_s)));

                xx=find(bias_pred==ori_base(k));
                p_s_bias(k)=length(xx)/length(bias_pred);
                b_s=histcounts2(tmp2(xx),temp2(xx),bins1,bins2,'Normalization','Probability');
                cond_ent_bias_pred(k)=nansum(nansum(b_s.*log2(b_s)));
            end
            condH_u_pred(i,j)=nansum(p_s_base.*cond_ent_base_pred);
            condH_b_pred(i,j)=nansum(p_s_bias.*cond_ent_bias_pred);

            % repeat conditional entropy for shuffled trials
    %         for k=1:length(ori_base)
    %             xx=find(ori_u_s(i,:)==ori_base(k));
    %             yy=find(ori_u_s(j,:)==ori_base(k));
    %             p_s_base(k)=length(xx)/length(base);
    %             u_s=histcounts2(us1(xx),us2(yy),bins1,bins2,'Normalization','Probability');
    %             cond_ent_base_shuf(k)=nansum(nansum(u_s.*log2(u_s)));
    %             
    %             xx=find(ori_b_s(i,:)==ori_base(k));
    %             yy=find(ori_b_s(j,:)==ori_base(k));
    %             p_s_bias(k)=length(xx)/length(bias);
    %             b_s=histcounts2(bs1(xx),bs2(yy),bins1,bins2,'Normalization','Probability');
    %             cond_ent_bias_shuf(k)=nansum(nansum(b_s.*log2(b_s)));
    %         end
    %         condH_u_shuf(i,j)=nansum(p_s_base.*cond_ent_base_shuf);
    %         condH_b_shuf(i,j)=nansum(p_s_bias.*cond_ent_bias_shuf);

            % calculate redundancy/MI: I(X,Y)=H(X)+H(Y)-H(X,Y)
            uMI(i,j)=(H_u(i,j)+condH_u(i,j))-(bb_sort(i)+bb_sort(j));
            bMI(i,j)=(H_b(i,j)+condH_b(i,j))-(aa_sort(i)+aa_sort(j));
%             uMI2(i,j)=uMI(i,j)/(H_base_sort(i)*H_base_sort(j));
%             bMI2(i,j)=bMI(i,j)/(H_bias_sort(i)*H_bias_sort(j));
%             uMI3(i,j)=uMI(i,j)/min([bb_sort(i) bb_sort(j)]);
%             bMI3(i,j)=bMI(i,j)/min([aa_sort(i) aa_sort(j)]);
            uMI_pred(i,j)=(H_u_pred(i,j)+condH_u_pred(i,j))-(bbp_sort(i)+bbp_sort(j));
            bMI_pred(i,j)=(H_b_pred(i,j)+condH_b_pred(i,j))-(aap_sort(i)+aap_sort(j));
%             uMI_pred2(i,j)=uMI_pred(i,j)/(H_basep_sort(i)*H_basep_sort(j));
%             bMI_pred2(i,j)=bMI_pred(i,j)/(H_biasp_sort(i)*H_biasp_sort(j));
%             uMI_pred3(i,j)=uMI_pred(i,j)/min([bbp_sort(i) bbp_sort(j)]);
%             bMI_pred3(i,j)=bMI_pred(i,j)/min([aap_sort(i) aap_sort(j)]);
    %         uMI_shuf(i,j)=(H_u_shuf(i,j)+condH_u_shuf(i,j))-(aa_sort(i)+aa_sort(j));
    %         bMI_shuf(i,j)=(H_b_shuf(i,j)+condH_b_shuf(i,j))-(bb_sort(i)+bb_sort(j));
        end
    end
    % reflect matrix:
    for i = 1:length(uMI)
        for j= 1:length(uMI)
            if i>j
                uMI(i,j)=uMI(j,i);
                bMI(i,j)=bMI(j,i);
                uMI_pred(i,j)=uMI_pred(j,i);
                bMI_pred(i,j)=bMI_pred(j,i);
%                 uMI2(i,j)=uMI2(j,i);
%                 bMI2(i,j)=bMI2(j,i);
%                 uMI_pred2(i,j)=uMI_pred2(j,i);
%                 bMI_pred2(i,j)=bMI_pred2(j,i);
%                 uMI3(i,j)=uMI3(j,i);
%                 bMI3(i,j)=bMI3(j,i);
%                 uMI_pred3(i,j)=uMI_pred3(j,i);
%                 bMI_pred3(i,j)=bMI_pred3(j,i);
    %             uMI_shuf(i,j)=uMI_shuf(j,i);
    %             bMI_shuf(i,j)=bMI_shuf(j,i);
            end
        end
    end
    clear a a2 a_xx a2_xx A_XX ans b b_pred B-s bins* con cond_* dir* i j index...
    k over* resp1 resp2 resp3 resp4 temp* tmp* u u_*

    cnv=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
    cnv=cnv/sum(cnv(:));
    save(name)
end
stop
% % plots:
cnv=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
cnv=cnv/sum(cnv(:));
% measured raw
frac3=100.*(bMI-uMI)./uMI;  
frac3(frac3>100)=100;
frac3(frac3<-100)=-100;
% predicted
frac4=100.*(bMI_pred-uMI_pred)./uMI_pred;
frac4(frac4>100)=100;
frac4(frac4<-100)=-100;
% normalized raw
frac5=100.*(bMI2-uMI2)./uMI2;
frac5(frac5>100)=100;
frac5(frac5<-100)=-100;
% normalized predicted
frac6=100.*(bMI_pred2-uMI_pred2)./uMI_pred2;
frac6(frac6>100)=100;
frac6(frac6<-100)=-100;

figure
supertitle('pairwise Redundancy: H(X)+H(Y)-H(X,Y)')
% % % % % % % % % % % % % % % % % % % % % % % % % % histograms maeasured:
subplot(4,4,1); box off; hold on
histogram(uMI)
plot(nanmean(uMI(:)),1000,'vk')
title('Redun uniform')
xlabel('Redun')
ylabel('proportion of cases (pairs)')
axis square
subplot(4,4,2); box off; hold on
histogram(bMI)
plot(nanmean(bMI(:)),1000,'vk')
title('Redun bias')
xlabel('Redun')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,3); box off; hold on
histogram(bMI-uMI)
plot(nanmean(nanmean(bMI-uMI)),1000,'vk')
title('Redun bias-uniform')
xlabel('change in Redun')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,4)
bins=(-100:10:100);
a3=histc(frac3(:),bins);
bar(bins+diff(bins(1:2))/2,a3/sum(a3),1);
hold on;box off
plot(nanmean(frac3(:)),0.45,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
xlabel('change in Redun (%)')
ylabel('Proportion of cases (pairs)')
title('% change in Redun')
axis square

subplot(4,4,5); box off; hold on
imagesc(conv2(uMI,cnv,'same'),[-0.5 0.5])
title('Redun uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,6); box off; hold on
imagesc(conv2(bMI,cnv,'same'),[-0.5 0.5])
title('Redun bias')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,7); box off; hold on
imagesc(conv2(bMI-uMI,cnv,'same'),[-0.5 0.5])
title('Redun bias-uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,8); box off
imagesc(conv2(frac3,cnv,'same'),[-100 100])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('% change in Redun')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out','YDir','normal')
axis square

subplot(4,4,9); box off; hold on
histogram(uMI_pred)
plot(nanmean(uMI_pred(:)),1000,'vk')
title('Redun uniform pred')
xlabel('Redun')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,10); box off; hold on
histogram(bMI_pred)
plot(nanmean(bMI_pred(:)),1000,'vk')
title('Redun bias pred')
xlabel('Redun')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,11); box off; hold on
histogram(bMI_pred-uMI_pred)
plot(nanmean(nanmean(bMI_pred-uMI_pred)),1000,'vk')
title('Redun bias-uniform pred')
xlabel('change in Redun')
ylabel('proportion of cases (pairs)')
axis square
subplot(4,4,12)
bins=(-50:5:50);
a4=histc(frac4(:),bins);
bar(bins+diff(bins(1:2))/2,a4/sum(a4),1);
hold on;box off
plot(nanmean(frac4(:)),0.55,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
xlabel('change in Redun (%)')
ylabel('Proportion of cases (pairs)')
title('Predicted % change in Redun')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')

subplot(4,4,13); box off; hold on
imagesc(conv2(uMI_pred,cnv,'same'),[-6 0])
title('Redun uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,14); box off; hold on
imagesc(conv2(bMI_pred,cnv,'same'),[-6 0])
title('Redun bias pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,15); box off; hold on
imagesc(conv2(bMI_pred-uMI_pred,cnv,'same'),[-.5 .5])
title('Redun bias-uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,16); box off
imagesc(conv2(frac4,cnv,'same'),[-50 50])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('Predicted % change in Redun')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out','YDir','normal')
axis square


% % % % % % % same for normalized values
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,1); box off; hold on
% histogram(uMI_norm,0:50)
% plot(nanmean(uMI_norm(:)),0.5,'vk')
% title('MI uniform normalized')
% xlabel('MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,2); box off; hold on
% histogram(bMI_norm,0:50)
% plot(nanmean(bMI_norm(:)),0.5,'vk')
% title('MI bias normalized')
% xlabel('MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,3); box off; hold on
% histogram(bMI_norm-uMI_norm)
% plot(nanmean(nanmean(bMI_norm-uMI_norm)),0.5,'vk')
% title('MI bias-uniform normalized')
% xlabel('change in MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,4)
% bins=(-100:10:100);
% a5=histc(frac5(:),bins);
% bar(bins+diff(bins(1:2))/2,a5/sum(a5),1);
% hold on;box off
% plot(nanmean(frac5(:)),0.45,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('% change in MI (normalized)')
% axis square
% % % % % % % % % % % % % % % % % % % % % % % % % imagesc matrices measured
% subplot(4,4,5); box off; hold on
% imagesc(conv2(uMI_norm,cnv,'same'))
% title('MI uniform norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,6); box off; hold on
% imagesc(conv2(bMI_norm,cnv,'same'))
% title('MI bias norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,7); box off; hold on
% imagesc(conv2(bMI_norm-uMI_norm,cnv,'same'))
% title('MI bias-uniform norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,8); box off
% imagesc(conv2(frac5,cnv,'same'),[-51 51])
% xlabel('unit 1 ori pref')
% ylabel('unit 2 ori pref')
% title('% change in MI (normalized)')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% axis square
% % % % % % % % % % % % % % % % % % % % % % 
% subplot(4,4,9); box off; hold on
% histogram(uMI_pred_norm)
% plot(nanmean(uMI_pred_norm(:)),0.5,'vk')
% title('MI uniform pred normalized')
% xlabel('MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,10); box off; hold on
% histogram(bMI_pred_norm)
% plot(nanmean(bMI_pred_norm(:)),0.5,'vk')
% title('MI bias pred normalized')
% xlabel('MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,11); box off; hold on
% histogram(bMI_pred_norm-uMI_pred_norm)
% plot(nanmean(nanmean(bMI_pred_norm-uMI_pred_norm)),0.5,'vk')
% title('MI bias-uniform pred normalized')
% xlabel('change in MI')
% ylabel('proportion of cases (pairs)')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,12)
% bins=(-100:10:100);
% a6=histc(frac6(:),bins);
% bar(bins+diff(bins(1:2))/2,a6/sum(a6),1);
% hold on;box off
% plot(nanmean(frac6(:)),0.45,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('predicted % change in MI (norm)')
% axis square

% % % % % % % % % % % % % % % % % % % % % % % % imagesc matrices predicted
% subplot(4,4,13); box off; hold on
% imagesc(conv2(uMI_pred_norm,cnv,'same'))
% title('MI uniform pred norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,14); box off; hold on
% imagesc(conv2(bMI_pred_norm,cnv,'same'))
% title('MI bias pred norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,8,15); box off; hold on
% imagesc(conv2(bMI_pred_norm-uMI_pred_norm,cnv,'same'))
% title('MI bias-uniform pred norm')
% xlabel('oripref unit 1')
% ylabel('oripref unit 2')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,16); box off
% imagesc(conv2(frac6,cnv,'same'),[-51 51])
% xlabel('unit 1 ori pref')
% ylabel('unit 2 ori pref')
% title('Predicted % change in MI (norm)')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% axis square
stopplots_MIacute
%%
% correlated - shuffled
figure
supertitle('deltaMI(correlated - shuffled)')
subplot(221); box off; hold on
histogram(uMI-uMI_shuf)
tmp=(uMI-uMI_shuf);
plot(nanmean(tmp(:)),800,'vk')
title('MI uniform')
xlabel('MI difference (reg-shuf)')
ylabel('proportion of cases (pairs)')
axis square
subplot(223); box off; hold on
histogram(bMI-bMI_shuf)
tmp=bMI-bMI_shuf;
plot(nanmean(tmp(:)),800,'vk')
title('MI bias')
xlabel('MI difference (reg-shuf)')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')

subplot(222); box off; hold on
imagesc(conv2(uMI-uMI_shuf,cnv,'same'),[0.01 0.6])
title('MI uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(224); box off; hold on
imagesc(conv2(bMI-bMI_shuf,cnv,'same'),[0.01 0.6])
title('MI bias')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')

stoppplots
%%
% old figure:
% subplot(221)
% frac3=100.*(bMI-uMI)./uMI;
% frac3(frac3>100)=100;
% frac3(frac3<-100)=-100;
% bins=(-100:10:100);
% a3=histc(frac3(:),bins);
% bar(bins+diff(bins(1:2))/2,a3/sum(a3),1);
% hold on;box off
% plot(nanmean(frac3(:)),0.45,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('Measured: % change in pairwise MI')
% axis square
% 
% subplot(222); box off
% imagesc(conv2(frac3,cnv,'same'),[-51 51])
% xlabel('unit 1 ori pref')
% ylabel('unit 2 ori pref')
% title('Measured: % change in pairwise MI')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% axis square
% 
% subplot(223)
% frac4=100.*(bMI_pred-uMI_pred)./uMI_pred;
% frac4(frac4>100)=100;
% frac4(frac4<-100)=-100;
% bins=(-30:5:30);
% a4=histc(frac4(:),bins);
% bar(bins+diff(bins(1:2))/2,a4/sum(a4),1);
% hold on;box off
% plot(nanmean(frac4(:)),0.55,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('Predicted: % change in pairwise MI')
% axis square
% 
% subplot(224); box off
% imagesc(conv2(frac4,cnv,'same'),[-21 21])
% xlabel('unit 1 ori pref')
% ylabel('unit 2 ori pref')
% title('Predicted: % change in pairwise MI')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% axis square

% % 
% subplot(132)
% oripref_bins=[168.75 11.25:22.5:168.75];
% for i=1:length(oripref_bins)-1
%     if i==1
%         tmp=frac2(find(oripref>oripref_bins(i)));
%         tmp2=frac2(find(oripref<oripref_bins(i+1)));
%         tmp=union(tmp,tmp2);
%     else
%         tmp=frac2(find(oripref>oripref_bins(i) & oripref<oripref_bins(i+1)));
%     end
%     dmi(i)=nanmean(tmp);
%     dmi_se(i)=nanstd(tmp)/length(tmp);
% end
% plotbins=(-90:22.5:90);
% errorline(plotbins(1:end-1),circshift(dmi,[0 4]),circshift(dmi_se,[0 4]),'k')
% hold on;box off
% plot((-90:90),zeros(181,1),':k')
% % plot([0 0],[-5 5],':r')
% xlabel('Oripref (deg)')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
% axis square
% ylabel('% change in MI')
% xlabel('Orientation preference (deg)')

% % % % % % 

%         % 1D probability matrices: P(x), P(y)
%         u1=histcounts(tmp,bins1,'Normalization','Probability');  %P(x) uniform
%         u2=histcounts(temp,bins2,'Normalization','Probability'); %P(y) uniform
%         b1=histcounts(tmp2,bins1,'Normalization','Probability'); %P(x) bias
%         b2=histcounts(temp2,bins2,'Normalization','Probability');%P(y) bias
%         
%         % pointwise Information
%         for m=1:size(u_pred,1)
%             for n=1:size(u_pred,2)
%                 tmpIu(m,n) = u_pred(m,n)*log2(u_pred(m,n)/(u1(m)*u2(n)));
%                 tmpIb(m,n) = b_pred(m,n)*log2(b_pred(m,n)/(b1(m)*b2(n)));
%             end
%         end
%         % sum pointwise above
%         I_u(i,j)=nansum(nansum(tmpIu));
%         I_b(i,j)=nansum(nansum(tmpIb));
% plots
% figure
% subplot(121)
% frac=100.*(I_b-I_u)./I_u;
% bins=(-100:10:100);
% a=histc(frac(:),bins);
% bar(bins+diff(bins(1:2))/2,a/sum(a),1);
% hold on;box off
% plot(mean(frac(:)),0.4,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1])%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases')
% 
% subplot(122)
% oripref_bins=[168.75 11.25:22.5:168.75];
% for i=1:length(oripref_bins)-1
%     if i==1
%         tmp=frac(find(oripref>oripref_bins(i)));
%         tmp2=frac(find(oripref<oripref_bins(i+1)));
%         tmp=union(tmp,tmp2);
%     else
%         tmp=frac(find(oripref>oripref_bins(i) & oripref<oripref_bins(i+1)));
%     end
%     dmi(i)=mean(tmp);
%     dmi_se(i)=std(tmp)/length(tmp);
% end
% plotbins=(-90:22.5:90);
% errorline(plotbins(1:end-1),circshift(dmi,[0 4]),circshift(dmi_se,[0 4]),'k')
% hold on;box off
% plot((-90:90),zeros(181,1),':k')
% % plot([0 0],[-5 5],':r')
% xlabel('Oripref (deg)')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
% ylabel('% change in MI')
% xlabel('Orientation preference (deg)')

