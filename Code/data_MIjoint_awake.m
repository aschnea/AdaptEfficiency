% load entropy files
clear

for n=[17 18 20 22 25 27]%15:27
    clearvars -except n
    if n==15
        load('cadetv1p366_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==16
        load('cadetv1p371_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==17
        load('cadetv1p384_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==18
        load('cadetv1p385_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==19
        load('cadetv1p392_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==20
        load('cadetv1p403_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==21
        load('cadetv1p419_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==22
        load('cadetv1p432_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==23
        load('cadetv1p437_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==24
        load('cadetv1p438_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==25
        load('cadetv1p460_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==26
        load('cadetv1p467_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    elseif n==27
        load('cadetv1p468_entropy','filename','aap_crct','bbp_crct','H_u_corrected','H_b_corrected','base_pred','bias_pred',...
            'resp_uniform','resp_bias','keep','keep2','min_count','oris','oripref_u','oribias_u')
    end
%% sort oris and matricies for work below
    [Y,I]=sort(oripref_u);
    % re-sort units by oripref; reorganize so 0/180 in the middle
%     tmp=abs(Y-90);
    tmp=find(Y==min(Y));
    Y2=[Y(tmp(1):end) Y(1:tmp(1))];
    I2=[I(tmp(1):end) I(1:tmp(1)-1)];
    resp_base_sort=resp_uniform(I2,:);
    resp_bias_sort=resp_bias(I2,:);
    oribias_sort=oribias_u(I2);

    aap_sort=aap_crct(I2);  % MI bias, corrected entropy
    bbp_sort=bbp_crct(I2);  % MI uniform, crctd entropy
    H_basep_sort=H_u_corrected(I2,:); % entropy uniform
    H_biasp_sort=H_b_corrected(I2,:); % entropy bias
    
    %only keep tuned cells:
    resp_base_sort=resp_base_sort(oribias_sort>0.3,:);
    resp_bias_sort=resp_bias_sort(oribias_sort>0.3,:);
    aap_sort=aap_sort(oribias_sort>0.3);
    bbp_sort=bbp_sort(oribias_sort>0.3);
    H_basep_sort=H_basep_sort(oribias_sort>0.3,:);
    H_biasp_sort=H_biasp_sort(oribias_sort>0.3,:);
    Y2=Y2(oribias_u>0.3);
% ignoring non-matched distributions and Stimulus domain
% only looking at redundancy of matched, entropy corrected, neuron domain
    
    %% pairwise MI
% %     % I(X,Y)=KL(P(x,y)|P(x)*P*y)) = I(X,Y)=sum_xy(P(x,y)*log2(P(x,y)/P(x)P(y)))
% %     % I(X,Y)=H(X)+H(Y)-H(X,Y)
% %     % measure redundancy as I_joint-(I_1+I_2).

    uMI_pred=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
    bR=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI_pred2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI_pred2=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     uMI_pred3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));
%     bMI_pred3=nan*zeros(size(resp_uniform,1),size(resp_uniform,1));

    % work for information:
    for i=1:size(resp_base_sort,1)-1        
        tmp=resp_base_sort(i,keep);  % for subsampling
        tmp2=resp_bias_sort(i,keep2);% for subsampling

        bins1=(0:1:max([tmp(:);tmp2(:)]));
        
        for j = i+1:size(resp_base_sort,1)
            temp=resp_base_sort(j,keep);
            temp2=resp_bias_sort(j,keep2);

            bins2=(0:1:max([temp(:);temp2(:)]));
            
            % 2D joint probability matrix: P(x,y)
            u_pred=histcounts2(tmp,temp,bins1,bins2,'Normalization','Probability');  %P(r) uniform
            b_pred=histcounts2(tmp2,temp2,bins1,bins2,'Normalization','Probability'); %P(r) bias
        
            % calculation joint entropy (H):
            H_u_pred(i,j)=-1*nansum(nansum(u_pred.*log2(u_pred)));
            H_b_pred(i,j)=-1*nansum(nansum(b_pred.*log2(b_pred)));

            % repeat conditional entropy for prediction (subsample)
            for k=1:length(oris)
                xx=find(base_pred==oris(k)); % # of trials of each ori
                u_s=histcounts2(tmp(xx),temp(xx),bins1,bins2,'Normalization','Probability');
                p_s_base(k)=length(xx)/length(base_pred); % ratio of given ori to total trials
                cond_ent_base_pred(k)=nansum(nansum(u_s.*log2(u_s)));
                
                xx=find(bias_pred==oris(k));
                b_s=histcounts2(tmp2(xx),temp2(xx),bins1,bins2,'Normalization','Probability');
                p_s_bias(k)=length(xx)/length(bias_pred);
                cond_ent_bias_pred(k)=nansum(nansum(b_s.*log2(b_s)));
            end
            condH_u_pred(i,j)=nansum(p_s_base.*cond_ent_base_pred);
            condH_b_pred(i,j)=nansum(p_s_bias.*cond_ent_bias_pred);
            
            % correct for finite data bias in entropy
            % Hexp=Htrue+sum_a(c_a/N^a) - c is weighting coefficient for a and
            % N is # of times each stimulus was repeated
            options = optimset('TolFun',1e-5,'TolX',1e-4,'Maxiter',10000,...
                'MaxFunEvals',10000,'Display','off','LargeScale','off');
            N=min_count;
            % Response Entropy:
            startu=[H_u_pred(i,j) 1 1 1];
            [Hj_u_corrected(i,j,:), error_uj(i,j)] = fminsearch(@(b) Hcorrection(b, H_u_pred(i,j), N),startu,options);
            startb=[H_b_pred(i,j) 1 1 1];
            [Hj_b_corrected(i,j,:), error_bj(i,j)] = fminsearch(@(b) Hcorrection(b, H_b_pred(i,j), N),startb,options);
            % conditional entropy (neuron domain):
            startu=[condH_u_pred(i,j) 1 1 1];
            [CondHj_u_corrected(i,j,:), error_Conduj(i,j)] = fminsearch(@(b) Hcorrection(b, condH_u_pred(i,j), N),startu,options);
            startb=[condH_b_pred(i,j) 1 1 1];
            [CondHj_b_corrected(i,j,:), error_Condbj(i,j)] = fminsearch(@(b) Hcorrection(b, condH_b_pred(i,j), N),startb,options);
            
            aapj_crct(i,j)=Hj_b_corrected(i,j,1)+CondHj_b_corrected(i,j,1);
            bbpj_crct(i,j)=Hj_u_corrected(i,j,1)+CondHj_u_corrected(i,j,1);
            
            % calculate redundancy/MI: I(X,Y)=H(X)+H(Y)-H(X,Y)
            uR(i,j)=bbpj_crct(i,j)-(bbp_sort(i)+bbp_sort(j));
            bR(i,j)=aapj_crct(i,j)-(aap_sort(i)+aap_sort(j));
%             uMI_pred(i,j)=(H_u_pred(i,j)+condH_u_pred(i,j))-(bbp_sort(i)+bbp_sort(j));
%             bMI_pred(i,j)=(H_b_pred(i,j)+condH_b_pred(i,j))-(aap_sort(i)+aap_sort(j));
%             uMI_pred2(i,j)=uMI_pred(i,j)/(H_basep_sort(i)*H_basep_sort(j));
%             bMI_pred2(i,j)=bMI_pred(i,j)/(H_biasp_sort(i)*H_biasp_sort(j));
%             uMI_pred3(i,j)=uMI_pred(i,j)/min([bbp_sort(i) bbp_sort(j)]);
%             bMI_pred3(i,j)=bMI_pred(i,j)/min([aap_sort(i) aap_sort(j)]);
            
%             clear u u1 u2 b b1 b2 u_s b_s p_* cond_* resp3 resp4...
%                 us1 bs1 us2 bs2 u_shuf b_shuf
        end
        clear resp1 resp2 tmp* temp*
    end

    % reflect matrix:
    for i = 1:length(uR)
        for j= 1:length(uR)
            if i>j
                uR(i,j)=uR(j,i);
                bR(i,j)=bR(j,i);
%                 uMI_pred2(i,j)=uMI_pred2(j,i);
%                 bMI_pred2(i,j)=bMI_pred2(j,i);
%                 uMI_pred3(i,j)=uMI_pred3(j,i);
%                 bMI_pred3(i,j)=bMI_pred3(j,i);
            end
        end
    end
    clear i j k tmp* temp* bins* xx u_s b_s u_pred b_pred cond_ent* start*
    
    savename=sprintf('%s_MIjoint_tune',filename);
    save(savename)
end
