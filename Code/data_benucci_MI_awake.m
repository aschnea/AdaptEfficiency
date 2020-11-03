%%  do mutual information for each channel/unit
clear
for n=15:27
    clearvars -except n
    if n==15
        load('cadetv1p366_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==16
        load('cadetv1p371_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==17
        load('cadetv1p384_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==18
        load('cadetv1p385_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==19
        load('cadetv1p392_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==20
        load('cadetv1p403_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==21
        load('cadetv1p419_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==22
        load('cadetv1p432_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==23
        load('cadetv1p437_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==24
        load('cadetv1p438_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==25
        load('cadetv1p460_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==26
        load('cadetv1p467_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    elseif n==27
        load('cadetv1p468_tuning','filename','oribias*','oripref*','tune*','oris','oris_b','oris_u','resp*','spont','stim_resp*')
    end

    %% identify responsive units
    for e=1:size(tune_u,1)
        if max(tune_u(e,1:9))>(spont(e,1)+3*spont(e,2))
            responsive(e)=1;
        else
            responsive(e)=0;
        end
    end
    responsive=logical(responsive);
    
    %% choose # of trials included
    % Reduced trials to look at MI power
%     tmp1=round(size(resp_uniform,2)/4); % /4 or /2
%     tmp2=round(size(resp_bias,2)/4);
%     resp_uniform=resp_uniform(responsive,end-tmp1:end);    
%     resp_bias=resp_bias(responsive,end-tmp2:end);
%     oris_u=oris_u(end-tmp1:end);
%     oris_b=oris_b(end-tmp2:end);
    
    % ALL TRIALS
    resp_uniform=resp_uniform(responsive,:);  
    resp_bias=resp_bias(responsive,:);    
    
    oripref_u=oripref_u(responsive);
    oribias_u=oribias_u(responsive);
    oripref_b=oripref_b(responsive);
    oribias_b=oribias_b(responsive);
    stim_resp_b=stim_resp_b{responsive,:};
    stim_resp_u=stim_resp_u{responsive,:};
    tune_u=tune_u(responsive,:);
    tune_b=tune_b(responsive,:);
    tune_sem_u=tune_sem_u(responsive,:);
    tune_sem_b=tune_sem_b(responsive,:);
    spont=spont(responsive,:);
    
    %% calculate entropy on full distributions
    for i=1:size(resp_uniform,1)
        tmp=resp_uniform(i,:);          % uniform response vector
        tmp2=resp_bias(i,:);            % bias response vector
        
        bins=(min([tmp(:);tmp2(:)]):1:max([tmp(:);tmp2(:)]));   % integer spike counts
        if length(bins)>length(tmp)/9
            disp('Should bin')
        end
        
        a=histc(tmp,bins)/length(tmp);      %P(r) uniform
        a2=histc(tmp2,bins)/length(tmp2);   %P(r) bias
        
        %   calculate response entropy
        for j=1:length(bins)
            entr_base(j)=a(j)*log2(a(j));
            entr_bias(j)=a2(j)*log2(a2(j));
        end
        %   calculate conditional entropy
        for j=1:length(oris)
            xx=find(oris_u==oris(j));     % cases of each ori
            a_xx=histc(tmp(xx),bins)/length(xx);    % P(r) in each bin for given ori
            p_s_base1(j)=length(xx)/length(oris_u);    % ratio of given ori to total trials
            for k=1:length(bins)
                cond_ent_base(j,k)=a_xx(k)*log2(a_xx(k));   % conditional entropy for uniform
            end
            
            xx=find(oris_b==oris(j));
            a2_xx=histc(tmp2(xx),bins);
            a2_xx=a2_xx/sum(a2_xx);
            p_s_bias1(j)=length(xx)/length(oris_b);
            for k=1:length(bins)
                cond_ent_bias(j,k)=a2_xx(k)*log2(a2_xx(k)); % conditional entropy for bias
            end
        end
%         H(X|Y)=sum( P(stim) * sum( P(r|stim)*log(P(r|stim)) ) )
%         P(stim)=p_s_base/bias
%         P(r|stim)=a_xx

        % Store Entropy and Conditional entropy in neuron (pref) domain
        H_base(i)=-1*nansum(entr_base); % measured entropy
        H_bias(i)=-1*nansum(entr_bias);
        CondH_base(i)=sum(p_s_base1.*nansum(cond_ent_base,2)');  % predicted entropy
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
        clear tmp* bins entr* cond*
    end
    %%  create matched distributions (uniform)
    
    %   find the biased ori
    a=histc(oris_b,oris);
    a2=histc(oris_u,oris);
    over_ori=oris(find(a==max(a)));
    over_ori_ind=find(a==max(a));
    over_rat=max(a)/min(a);
    
    %   make a biased response distribution in the base condition
%     keep=find(oris_u==over_ori);
%     keep=keep(:);
%     keep2=find(oris_b==over_ori);
%     keep2=keep2(randperm(length(keep2),length(keep)))';
%     target=round(length(keep)/over_rat);
%     for i=1:length(oris)
%         if i~=over_ori_ind
%             tmp=find(oris_u==oris(i));
%             keep=[keep;tmp(randperm(target))'];
%             %this the the subset of trials to keep in uniform to get the same ori dist as biased
%             temp=find(oris_b==oris(i));
%             keep2=[keep2;temp(randperm(target))'];
%         end
%     end
    
    %   make uniform distribution in both environments
    keep=[];
    keep2=[];
    min_count=min([a a2]);   % Trials of each ori to create distribution
    for i=1:length(oris)
        % trials to keep from uniform
        tmp=find(oris_u==oris(i));
        keep=[keep;tmp(randperm(min_count))'];
        % trials to keep from bias
        temp=find(oris_b==oris(i));
        keep2=[keep2;temp(randperm(min_count))'];
    end
    
    %%   compute MI for these matched distributions
    for i=1:size(resp_uniform,1)
        tmp=resp_uniform(i,keep);
        tmp2=resp_bias(i,keep2);
        base_pred=oris_u(keep);       %kept stimuli
        bias_pred=oris_b(keep2);
        bins=(min([tmp(:);tmp2(:)]):1:max([tmp(:);tmp2(:)]));
        if length(bins)>length(tmp)/9
            disp('Should bin')
        end
        a=histc(tmp,bins)/length(tmp);      %P(r)
        a2=histc(tmp2,bins)/length(tmp2);
        % response entropy: 
        for j=1:length(bins)                %entropy of responses
            entr_base(j)=a(j)*log2(a(j));
            entr_bias(j)=a2(j)*log2(a2(j));
        end
        % conditional entropy:
        for j=1:length(oris)
            xx=find(base_pred==oris(j)); %find index of each ori
            a_xx=histc(tmp(xx),bins)/length(xx); %P(r) for each ori
            p_s_base2(j)=length(xx)/length(base_pred); %P(ori)
            for k=1:length(bins)
                cond_ent_base(j,k)=a_xx(k)*log2(a_xx(k));
            end
            
            xx=find(bias_pred==oris(j));
            a2_xx=histc(tmp2(xx),bins);
            a2_xx=a2_xx/sum(a2_xx);
            p_s_bias2(j)=length(xx)/length(bias_pred);
            for k=1:length(bins)
                cond_ent_bias(j,k)=a2_xx(k)*log2(a2_xx(k));
            end
        end        
        
        % Store Entropy and Conditional entropy in neuron (pref) domain
        H_base_pred(i)=-1*nansum(entr_base); % measured entropy
        H_bias_pred(i)=-1*nansum(entr_bias);
        CondH_base_pred(i)=sum(p_s_base2.*nansum(cond_ent_base,2)');% predicted entropy
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
        for j = 1:length(oris)
            startu=[CondH_base_stim_pred(i,j) 1 1 1];
            [CondH_u_corrected_stim(i,j,:), error_Condu(i,j)] = fminsearch(@(b) Hcorrection(b, CondH_base_stim_pred(i,j), N),startu,options);
            startb=[CondH_bias_stim_pred(i,j) 1 1 1];
            [CondH_b_corrected_stim(i,j,:), error_Condb(i,j)] = fminsearch(@(b) Hcorrection(b, CondH_bias_stim_pred(i,j), N),startb,options);
        end
        
        
        aap_crct(i)=H_b_corrected(i,1)+CondH_b_corrected(i,1);
        bbp_crct(i)=H_u_corrected(i,1)+CondH_u_corrected(i,1);
        aap_stim_crct(i,:)=H_b_corrected(i,1)+squeeze(CondH_b_corrected_stim(i,:,1));
        bbp_stim_crct(i,:)=H_u_corrected(i,1)+squeeze(CondH_u_corrected_stim(i,:,1));
        clear tmp* bins entr* cond* start*
    end
%     figure; subplot(221); plot(aap_crct,aap,'.'); refline(1,0); 
%     subplot(222); plot(bbp_crct,bbp,'.'); refline(1,0); 
%     subplot(223); plot(bbp,aap,'.'); refline(1,0); 
%     subplot(224);plot(bbp_crct,aap_crct,'.');refline(1,0)


    clear a a2 a_xx a2_xx ans b b_pred B* bins* con cond_* dir* i j index...
    k over* resp1 resp2 resp3 resp4 temp* tmp* u u_*
    savename=sprintf('%s_entropy',filename);
    save(savename)
end
stopcalc

%     %% shuffle responses for I_shuffle
%     % % % % shuffled Information only useful for single class/ori. Ignoring it
% 
% %     for i = 1:size(resp_bias,1)
% %         shuf=randperm(size(resp_uniform,2));
% %         shuf2=randperm(size(resp_bias,2));
% %         shuf_u(i,:)=resp_uniform(i,shuf);
% %         shuf_b(i,:)=resp_bias(i,:);
% %         ori_u_s(i,:)=oris_u(shuf); % keep each list of orientations for CondH
% %         ori_b_s(i,:)=oris_b(shuf2);
% %     end
%     %% sort oris and matricies for work below
%     [Y,I]=sort(oripref_u);
%     % re-sort units by oripref; reorganize so 0/180 in the middle
%     tmp=abs(Y-90);
%     tmp=find(tmp==min(tmp));
%     Y2=[Y(tmp(1):end) Y(1:tmp(1))];
%     I2=[I(tmp(1):end) I(1:tmp(1)-1)];
%     resp_base_sort=resp_uniform(I2,:);
%     resp_bias_sort=resp_bias(I2,:);
% %     shuf_u_sort=shuf_u(I2,:);
% %     shuf_b_sort=shuf_b(I2,:);
%     aa_sort=aa(I2);
%     bb_sort=bb(I2);
%     aap_sort=aap(I2);
%     bbp_sort=bbp(I2);
%     H_base_sort=H_base(I2);
%     H_bias_sort=H_bias(I2);
%     H_basep_sort=H_base_pred(I2);
%     H_biasp_sort=H_bias_pred(I2);
%     
%     %% pairwise MI
% %     % I(X,Y)=KL(P(x,y)|P(x)*P*y)) = I(X,Y)=sum_xy(P(x,y)*log2(P(x,y)/P(x)P(y)))
% %     % I(X,Y)=H(X)+H(Y)-H(X,Y)
% %     % measure redundancy as I_joint-(I_1+I_2).
%     uMI=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI_pred=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI_pred=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI_pred2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI_pred2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI_pred3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI_pred3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
% % %     uMI_shuf=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
% % %     bMI_shuf=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
% % 
%     % work for information:
%     for i=1:size(resp_base_sort,1)-1        
%         tmp=resp_base_sort(i,keep);  % for subsampling
%         tmp2=resp_bias_sort(i,keep2);% for subsampling
%         resp1=resp_base_sort(i,:);   % for full experiment
%         resp2=resp_bias_sort(i,:);
% %         us1=shuf_u_sort(i,:);
% %         bs1=shuf_b_sort(i,:);
%         bins1=(0:1:max([resp1(:);resp2(:)]));
%         
% %         if length(bins1)>length(tmp)/9
% %             disp('Should bin')
% %         end
%         
%         for j = i+1:size(resp_base_sort,1)
%             temp=resp_base_sort(j,keep);
%             temp2=resp_bias_sort(j,keep2);
%             resp3=resp_base_sort(j,:);
%             resp4=resp_bias_sort(j,:);
% %             us2=shuf_u_sort(j,:);
% %             bs2=shuf_b_sort(j,:);
%             bins2=(0:1:max([resp3(:);resp4(:)]));
%             
%             % 2D joint probability matrix: P(x,y)
%             u_pred=histcounts2(tmp,temp,bins1,bins2,'Normalization','Probability');  %P(r) uniform
%             b_pred=histcounts2(tmp2,temp2,bins1,bins2,'Normalization','Probability'); %P(r) bias
%             u=histcounts2(resp1,resp3,bins1,bins2,'Normalization','Probability');  %P(r) uniform
%             b=histcounts2(resp2,resp4,bins1,bins2,'Normalization','Probability'); %P(r) bias
% %             u_shuf=histcounts2(us1,us2,bins1,bins2,'Normalization','Probability');
% %             b_shuf=histcounts2(bs1,bs2,bins1,bins2,'Normalization','Probability');
%         
%             % calculation joint entropy (H):
%             H_u(i,j)=-1*nansum(nansum(u.*log2(u)));
%             H_b(i,j)=-1*nansum(nansum(b.*log2(b)));
%             H_u_pred(i,j)=-1*nansum(nansum(u_pred.*log2(u_pred)));
%             H_b_pred(i,j)=-1*nansum(nansum(b_pred.*log2(b_pred)));
% %             H_u_shuf(i,j)=-1*nansum(nansum(u_shuf.*log2(u_shuf)));
% %             H_b_shuf(i,j)=-1*nansum(nansum(b_shuf.*log2(b_shuf)));
%             
%             % calculate conditional joint entropy (condH):
%             for k=1:length(oris)
%                 xx=find(oris_u==oris(k)); % # of trials of each ori
%                 p_s_base(k)=length(xx)/length(oris_u); % ratio of given ori to total trials
%                 % joint probability matrix for stimulus k for units i, j:
%                 u_s=histcounts2(resp1(xx),resp3(xx),bins1,bins2,'Normalization','Probability');
%                 cond_ent_base(k)=nansum(nansum(u_s.*log2(u_s)));
%                 
%                 xx=find(oris_b==oris(k));
%                 p_s_bias(k)=length(xx)/length(oris_b);
%                 b_s=histcounts2(resp2(xx),resp4(xx),bins1,bins2,'Normalization','Probability');
%                 cond_ent_bias(k)=nansum(nansum(b_s.*log2(b_s)));
%             end
%             condH_u(i,j)=nansum(p_s_base.*cond_ent_base);
%             condH_b(i,j)=nansum(p_s_bias.*cond_ent_bias);
%             
%             % repeat conditional entropy for prediction (subsample)
%             for k=1:length(oris)
%                 xx=find(base_pred==oris(k)); % # of trials of each ori
%                 p_s_base(k)=length(xx)/length(base_pred); % ratio of given ori to total trials
%                 u_s=histcounts2(tmp(xx),temp(xx),bins1,bins2,'Normalization','Probability');
%                 cond_ent_base_pred(k)=nansum(nansum(u_s.*log2(u_s)));
%                 
%                 xx=find(bias_pred==oris(k));
%                 p_s_bias(k)=length(xx)/length(bias_pred);
%                 b_s=histcounts2(tmp2(xx),temp2(xx),bins1,bins2,'Normalization','Probability');
%                 cond_ent_bias_pred(k)=nansum(nansum(b_s.*log2(b_s)));
%             end
%             condH_u_pred(i,j)=nansum(p_s_base.*cond_ent_base_pred);
%             condH_b_pred(i,j)=nansum(p_s_bias.*cond_ent_bias_pred);
%             
%             % repeat conditional entropy for shuffled trials
% %             for k=1:length(oris)
% %                 xx=find(ori_u_s(i,:)==oris(k));
% %                 yy=find(ori_u_s(j,:)==oris(k));
% %                 p_s_base(k)=length(xx)/length(oris_u);
% %                 u_s=histcounts2(us1(xx),us2(yy),bins1,bins2,'Normalization','Probability');
% %                 cond_ent_base_shuf(k)=nansum(nansum(u_s.*log2(u_s)));
% %                 
% %                 xx=find(ori_b_s(i,:)==oris(k));
% %                 yy=find(ori_b_s(j,:)==oris(k));
% %                 p_s_bias(k)=length(xx)/length(oris_b);
% %                 b_s=histcounts2(bs1(xx),bs2(yy),bins1,bins2,'Normalization','Probability');
% %                 cond_ent_bias_shuf(k)=nansum(nansum(b_s.*log2(b_s)));
% %             end
% %             condH_u_shuf(i,j)=nansum(p_s_base.*cond_ent_base_shuf);
% %             condH_b_shuf(i,j)=nansum(p_s_bias.*cond_ent_bias_shuf);
%             
%             % calculate redundancy/MI: I(X,Y)=H(X)+H(Y)-H(X,Y)
%             uMI(i,j)=(H_u(i,j)+condH_u(i,j))-(bb_sort(i)+bb_sort(j));
%             bMI(i,j)=(H_b(i,j)+condH_b(i,j))-(aa_sort(i)+aa_sort(j));
%             uMI2(i,j)=uMI(i,j)/(H_base_sort(i)*H_base_sort(j));
%             bMI2(i,j)=bMI(i,j)/(H_bias_sort(i)*H_bias_sort(j));
%             uMI3(i,j)=uMI(i,j)/min([bb_sort(i) bb_sort(j)]);
%             bMI3(i,j)=bMI(i,j)/min([aa_sort(i) aa_sort(j)]);
%             uMI_pred(i,j)=(H_u_pred(i,j)+condH_u_pred(i,j))-(bbp_sort(i)+bbp_sort(j));
%             bMI_pred(i,j)=(H_b_pred(i,j)+condH_b_pred(i,j))-(aap_sort(i)+aap_sort(j));
%             uMI_pred2(i,j)=uMI_pred(i,j)/(H_basep_sort(i)*H_basep_sort(j));
%             bMI_pred2(i,j)=bMI_pred(i,j)/(H_biasp_sort(i)*H_biasp_sort(j));
%             uMI_pred3(i,j)=uMI_pred(i,j)/min([bbp_sort(i) bbp_sort(j)]);
%             bMI_pred3(i,j)=bMI_pred(i,j)/min([aap_sort(i) aap_sort(j)]);
% %             uMI_shuf(i,j)=(H_u_shuf(i,j)+condH_u_shuf(i,j))-(aa_sort(i)+aa_sort(j));
% %             bMI_shuf(i,j)=(H_b_shuf(i,j)+condH_b_shuf(i,j))-(bb_sort(i)+bb_sort(j));
%             
%             clear u u1 u2 b b1 b2 u_s b_s p_* cond_* resp3 resp4...
%                 us1 bs1 us2 bs2 u_shuf b_shuf
%         end
%         clear resp1 resp2 tmp*
%     end
%     % reflect matrix:
%     for i = 1:length(uMI)
%         for j= 1:length(uMI)
%             if i>j
%                 uMI(i,j)=uMI(j,i);
%                 bMI(i,j)=bMI(j,i);
%                 uMI_pred(i,j)=uMI_pred(j,i);
%                 bMI_pred(i,j)=bMI_pred(j,i);
%                 uMI2(i,j)=uMI2(j,i);
%                 bMI2(i,j)=bMI2(j,i);
%                 uMI_pred2(i,j)=uMI_pred2(j,i);
%                 bMI_pred2(i,j)=bMI_pred2(j,i);
%                 uMI3(i,j)=uMI3(j,i);
%                 bMI3(i,j)=bMI3(j,i);
%                 uMI_pred3(i,j)=uMI_pred3(j,i);
%                 bMI_pred3(i,j)=bMI_pred3(j,i);
% %                 uMI_shuf(i,j)=uMI_shuf(j,i);
% %                 bMI_shuf(i,j)=bMI_shuf(j,i);
%             end
%         end
%     end
%% pairwise plots:
cnv=[0.1 0.3 0.1;0.3 0.5 0.3;0.1 0.3 0.1];
cnv=cnv/sum(cnv(:));
% 2:1 measured raw
frac3=100.*(bMI2_avg-uMI2_avg)./uMI2_avg;  
frac3(frac3>100)=100;
frac3(frac3<-100)=-100;
% 2:1 predicted
frac4=100.*(bMI2_pred_avg-uMI2_pred_avg)./uMI2_pred_avg;
frac4(frac4>100)=100;
frac4(frac4<-100)=-100;
% 6:1 measured
frac5=100.*(bMI6_avg-uMI6_avg)./uMI6_avg;
frac5(frac5>100)=100;
frac5(frac5<-100)=-100;
% 6:1 predicted
frac6=100.*(bMI6_pred_avg-uMI6_pred_avg)./uMI6_pred_avg;
frac6(frac6>100)=100;
frac6(frac6<-100)=-100;


figure
supertitle('Awake, 2:1, redundancy')
% % % % % % % % % % % % % % % % % % % % % % % % % % histograms maeasured:
subplot(4,4,1); box off; hold on
histogram(uMI2)
plot(nanmean(uMI2(:)),2000,'vk')
title('MI uniform')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
subplot(4,4,2); box off; hold on
histogram(bMI2)
plot(nanmean(bMI2(:)),2000,'vk')
title('MI bias')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,3); box off; hold on
histogram(bMI2-uMI2)
plot(nanmean(nanmean(bMI2-uMI2)),2000,'vk')
title('MI bias-uniform')
xlabel('change in MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,4)
% bins=(-100:10:100);
% a3=histc(frac3(:),bins);
% bar(bins+diff(bins(1:2))/2,a3/sum(a3),1);
% hold on;box off
% plot(nanmean(frac3(:)),0.45,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('% change in MI')
% axis square

subplot(4,4,5); box off; hold on
imagesc(conv2(uMI2_avg,cnv,'same'),[0 0.25])
title('MI uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,6); box off; hold on
imagesc(conv2(bMI2_avg,cnv,'same'),[0 0.25])
title('MI bias')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,7); box off; hold on
imagesc(conv2(bMI2_avg-uMI2_avg,cnv,'same'),[-0.1 0.1])
title('MI bias-uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,8); box off
imagesc(conv2(frac3,cnv,'same'),[-20 20])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('% change in MI')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
axis square

subplot(4,4,9); box off; hold on
histogram(uMI2_pred)
plot(nanmean(uMI2_pred(:)),2000,'vk')
title('MI uniform pred')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,10); box off; hold on
histogram(bMI2_pred)
plot(nanmean(bMI2_pred(:)),2000,'vk')
title('MI bias pred')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,11); box off; hold on
histogram(bMI2_pred-uMI2_pred)
plot(nanmean(nanmean(bMI2_pred-uMI2_pred)),2000,'vk')
title('MI bias-uniform pred')
xlabel('change in MI')
ylabel('proportion of cases (pairs)')
axis square
% subplot(4,4,12)
% bins=(-30:5:30);
% a4=histc(frac4(:),bins);
% bar(bins+diff(bins(1:2))/2,a4/sum(a4),1);
% hold on;box off
% plot(nanmean(frac4(:)),0.55,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('Predicted % change in MI')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')

subplot(4,4,13); box off; hold on
imagesc(conv2(uMI2_pred_avg,cnv,'same'),[0 0.25])
title('MI uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,14); box off; hold on
imagesc(conv2(bMI2_pred_avg,cnv,'same'),[0 0.25])
title('MI bias pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,15); box off; hold on
imagesc(conv2(bMI2_pred_avg-uMI2_pred_avg,cnv,'same'),[-.25 0])
title('MI bias-uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,16); box off
imagesc(conv2(frac4,cnv,'same'),[-55 25])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('Predicted % change in MI')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
axis square

% % % % % 
figure
supertitle('Awake, 6:1, redundancy')
% % % % % % % % % % % % % % % % % % % % % % % % % % histograms maeasured:
subplot(4,4,1); box off; hold on
histogram(uMI6)
plot(nanmean(uMI6(:)),2000,'vk')
title('MI uniform')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
subplot(4,4,2); box off; hold on
histogram(bMI6)
plot(nanmean(bMI6(:)),2000,'vk')
title('MI bias')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,3); box off; hold on
histogram(bMI6-uMI6)
plot(nanmean(nanmean(bMI6-uMI6)),2000,'vk')
title('MI bias-uniform')
xlabel('change in MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
% subplot(4,4,4)
% bins=(-100:10:100);
% a3=histc(frac3(:),bins);
% bar(bins+diff(bins(1:2))/2,a3/sum(a3),1);
% hold on;box off
% plot(nanmean(frac3(:)),0.45,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('% change in MI')
% axis square

subplot(4,4,5); box off; hold on
imagesc(conv2(uMI6_avg,cnv,'same'),[0 0.2])
title('MI uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,6); box off; hold on
imagesc(conv2(bMI6_avg,cnv,'same'),[0 0.2])
title('MI bias')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,7); box off; hold on
imagesc(conv2(bMI6_avg-uMI6_avg,cnv,'same'),[-0.1 0])
title('MI bias-uniform')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,8); box off
imagesc(conv2(frac5,cnv,'same'),[-100 0])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('% change in MI')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
axis square

subplot(4,4,9); box off; hold on
histogram(uMI6_pred)
plot(nanmean(uMI6_pred(:)),2000,'vk')
title('MI uniform pred')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,10); box off; hold on
histogram(bMI6_pred)
plot(nanmean(bMI6_pred(:)),2000,'vk')
title('MI bias pred')
xlabel('MI')
ylabel('proportion of cases (pairs)')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,11); box off; hold on
histogram(bMI6_pred-uMI6_pred)
plot(nanmean(nanmean(bMI6_pred-uMI6_pred)),2000,'vk')
title('MI bias-uniform pred')
xlabel('change in MI')
ylabel('proportion of cases (pairs)')
axis square
% subplot(4,4,12)
% bins=(-30:5:30);
% a4=histc(frac4(:),bins);
% bar(bins+diff(bins(1:2))/2,a4/sum(a4),1);
% hold on;box off
% plot(nanmean(frac4(:)),0.55,'vk')
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')%,'XLim',[-12.5 12.5])
% xlabel('Change in MI (%)')
% ylabel('Proportion of cases (pairs)')
% title('Predicted % change in MI')
% axis square
% set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')

subplot(4,4,13); box off; hold on
imagesc(conv2(uMI6_pred_avg,cnv,'same'),[-.4 0.4])
title('MI uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,14); box off; hold on
imagesc(conv2(bMI6_pred_avg,cnv,'same'),[-.4 0.4])
title('MI bias pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,15); box off; hold on
imagesc(conv2(bMI6_pred_avg-uMI6_pred_avg,cnv,'same'),[-.75 0])
title('MI bias-uniform pred')
xlabel('oripref unit 1')
ylabel('oripref unit 2')
axis square
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
subplot(4,4,16); box off
imagesc(conv2(frac6,cnv,'same'),[-100 0])
xlabel('unit 1 ori pref')
ylabel('unit 2 ori pref')
title('Predicted % change in MI')
set(gca,'PlotBoxAspectRatio',[3 2 1],'TickDir','out')
axis square