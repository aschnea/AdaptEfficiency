%% bennuci_GLM - acute
clear

for n=33%[19 22 25 30 33]%1:33%1:17
    clearvars -except n
    if n==1
        load('129r001p173_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='129r001p173_glm';
    elseif n==2
        load('130l001p169_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='130l001p169_glm';
    elseif n==3
        load('140l001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p107_glm';
    elseif n==4
        load('140l001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p122_glm';
    elseif n==5
        load('140r001p105_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p105_glm';
    elseif n==6
        load('140r001p122_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p122_glm';
    elseif n==7
        load('130l001p170_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='130l001p170_glm';
    elseif n==8
        load('140l001p108_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p108_glm';
    elseif n==9
        load('140l001p110_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l001p110_glm';
    elseif n==10
        load('140r001p107_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p107_glm';
    elseif n==11
        load('140r001p109_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r001p109_glm';
    elseif n==12
        load('lowcon114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon114_glm';
    elseif n==13
        load('lowcon115_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon115_glm';
    elseif n==14
        load('lowcon116_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon116_glm';
    elseif n==15
        load('lowcon117_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='lowcon117_glm';
    elseif n==16
        load('140l113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140l113_awaketime_glm';
    elseif n==17
        load('140r113_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias');
        name='140r113_awaketime_glm';
    elseif n==18 % start of experiment 141 files
        load('141r001p006_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p006_awaketime_glm';
    elseif n==19
        load('141r001p007_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p007_awaketime6_glm';
    elseif n==20
        load('141r001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p009_awaketime_fine_glm';
    elseif n==21 % rotated AT 4:1 (80°)
        load('141r001p024_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p024_awaketime_glm';
    elseif n==22 % rotated AT 6:1 (80°)
        load('141r001p025_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p025_awaketime6_glm';
    elseif n==23 % rotated AT fineori (90°??)
        load('141r001p027_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p027_awaketime_fine';
    elseif n==24 % rotated fineori (40°)
        load('141r001p038_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p038_awaketime_fine';
    elseif n==25 % rotated 6:1 (120°)
        load('141r001p039_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p039_awaketime6_glm';
    elseif n==26 % rotated awaketime 4:1 (120°)
        load('141r001p041_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p041_awaketime_glm';
    elseif n==27
        load('141r001p114_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='141r001p114_glm';
        
    elseif n==28
        load('142l001p002_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p002_awaketime_glm';
    elseif n==29
        load('142l001p004_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p004_awaketime_fine_glm';
    elseif n==30
        load('142l001p006_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p006_awaketime6_glm';
    elseif n==31
        load('142l001p007_awaketime_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p007_awaketime_glm';
    elseif n==32
        load('142l001p009_awaketime_fine_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p009_awaketime_fine_glm';
    elseif n==33
        load('142l001p010_awaketime6_entropy','responsive','keep*','ori*','resp*','spont*','base','bias')
        name='142l001p010_awaketime6_glm';
    end
    
    num_units=sum(responsive);
    val_mode=mode(bias);    % identify bias ori
    if val_mode==80         % align biased ori to first column
        shiftori=1;
    else
        shiftori=0;
    end
    
    clearvars -except n ori_base base bias oripref oribias resp_raw* filename shiftori name
    % remove blanks trials
    [blank1]=find(base~=200);
    [blank2]=find(bias~=200);
    base2=base(blank1); % Uniform trial oris w/o blanks
    bias2=bias(blank2); % Bias trial oris w/o blanks
    resp_raw_base=resp_raw_base(:,blank1);
    resp_raw_bias=resp_raw_bias(:,blank2);
    
    % identify oris on each trial
    ori_trialu=zeros(length(ori_base)-1,size(resp_raw_base,2));
    ori_trialb=zeros(length(ori_base)-1,size(resp_raw_bias,2));
    for i=1:length(ori_base)-1
        [~,x]=find(base==ori_base(i));
        [~,y]=find(bias==ori_base(i));
        Nstimu(i)=length(x);
        Nstimb(i)=length(y);
        ori_trialu(i,x)=1;
        ori_trialb(i,y)=1;
    end
    ori_trialu=ori_trialu(:,blank1);
    ori_trialb=ori_trialb(:,blank2);
    ori_trialu=logical(ori_trialu);
    ori_trialb=logical(ori_trialb);
    
    %% GLM on all units
%     % set GLM inputs
%     spku=resp_raw_base;
%     spkb=resp_raw_bias;
%     vsort=randperm(size(spku,1));
% %     Ns=Nstimu; spk=spku; u_oris=ori_base(1:end-1); idxs=ori_trialu; test_ori=0; vsort=1:10;
% %     %69 IS LOWEST UNIT COUNT IN ALL ACUTE FILES. %30 is what I use in LDA neighbors
% 
%     for i = 1:length(ori_base)-1
%         test_ori=ori_base(i);
%         
%         [erroru(i,:,:,:), pred_u(i,:,:,:), proj_u(i,:,:,:), GLMori(i,:)]=TenFold_1toN_sens(Nstimu,spku,ori_base(1:end-1),test_ori,ori_trialu,vsort);
%         [errorb(i,:,:,:), pred_b(i,:,:,:), proj_b(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,ori_base(1:end-1),test_ori,ori_trialb,vsort);
%         % 9 test oris x 10 units x 8 ori pars x 10 repeats
%         % need to resort bc test ori gets moved to first index 1st
%         % dimension each time (see GLMori)
%         disp('next test ori')
%     end
% %     stop
%     save(name,'erroru','errorb','pred_u','pred_b','proj_u','proj_b','GLMori',...
%         'ori_base','oripref','oribias','shiftori','vsort')
    %% separate OSI groups 

% plot change in decoder or d' as a function of OSI. Fold orientations (0
% 40 80)
    
    % fixed bins of equal length
    fixed_OSI=0.01:0.33:1;
    for b=1:length(fixed_OSI)-1
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=find(oribias>=fixed_OSI(b) & oribias<=fixed_OSI(b+1));
        n_units_f=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_raw_base(osi_keep,:);
        spkb=resp_raw_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(ori_base)-1
            test_ori=ori_base(i);
            
            [erroru_f(i,:,:,:), pred_u_f(i,:,:,:), proj_u_f(i,:,:,:), GLMori_f(i,:)]=TenFold_1toN_sens(Nstimu,spku,ori_base(1:end-1),test_ori,ori_trialu,vsort);
            [errorb_f(i,:,:,:), pred_b_f(i,:,:,:), proj_b_f(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,ori_base(1:end-1),test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_f%d',name,b),'erroru_f','errorb_f','pred_u_f','pred_b_f','proj_u_f','proj_b_f','GLMori_f',...
            'ori_base','oripref','oribias','shiftori','n_units_f','fixed_OSI')
    end
    
    % sliding bin of equal length w/ overlap
    slide_OSI=[0.1 0.4; 0.2 0.5; 0.3 0.6; 0.4 0.7; 0.5 0.8; 0.6 0.9; 0.7 1];
    for b=1:size(slide_OSI,1)
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=find(oribias>=slide_OSI(b,1) & oribias<=slide_OSI(b,2));
        n_units_s=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_raw_base(osi_keep,:);
        spkb=resp_raw_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(ori_base)-1
            test_ori=ori_base(i);
            
            [erroru_s(i,:,:,:), pred_u_s(i,:,:,:), proj_u_s(i,:,:,:), GLMori_s(i,:)]=TenFold_1toN_sens(Nstimu,spku,ori_base(1:end-1),test_ori,ori_trialu,vsort);
            [errorb_s(i,:,:,:), pred_b_s(i,:,:,:), proj_b_s(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,ori_base(1:end-1),test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_s%d',name,b),'erroru_s','errorb_s','pred_u_s','pred_b_s','proj_u_s','proj_b_s','GLMori_s',...
            'ori_base','oripref','oribias','shiftori','n_units_s','slide_OSI')
    end
    
    % same # of units in each group but variable bin sizes
    [R,I]=sort(oribias);
    L=round(linspace(1,length(oribias),4));

    for b=1:length(L)-1
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=(I(L(b):L(b+1)));
        n_units_p=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_raw_base(osi_keep,:);
        spkb=resp_raw_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(ori_base)-1
            test_ori=ori_base(i);
            
            [erroru_p(i,:,:,:), pred_u_p(i,:,:,:), proj_u_p(i,:,:,:), GLMori_p(i,:)]=TenFold_1toN_sens(Nstimu,spku,ori_base(1:end-1),test_ori,ori_trialu,vsort);
            [errorb_p(i,:,:,:), pred_b_p(i,:,:,:), proj_b_p(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,ori_base(1:end-1),test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_p%d',name,b),'erroru_p','errorb_p','pred_u_p','pred_b_p','proj_u_p','proj_b_p','GLMori_p',...
            'ori_base','oripref','oribias','shiftori','n_units_p','L')
    end
    
    %% save
    clear ans i min_trial tmp tmp2 x y
    disp(name)
%     save(name)
end
stop
tmp=squeeze(mean(erroru(1,:,:,:),4));
tmp2=squeeze(mean(errorb(1,:,:,:),4));
figure; hold on; plot(mean(tmp,1)); plot(mean(tmp2,1))


%% GLM - Awake
clear
oris=0:20:160;

for a=[17 18 20 22 25 27] % 6:1 files % 26:27
    clearvars -except a oris
    if a==1
        load('cadetv1p194_corr','filename','resp_b_sub','resp_u_sub','keep','keep2');
        load('cadetv1p194_tuning','ori*','resp*','spont','tune*');
    elseif a==15
%         load('cadetv1p366_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p366_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==16
%         load('cadetv1p371_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p371_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==17
%         load('cadetv1p384_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p384_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');;
    elseif a==18
%         load('cadetv1p385_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p385_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==19
%         load('cadetv1p392_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p392_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==20
%         load('cadetv1p403_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p403_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==21
%         load('cadetv1p419_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p419_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==22
%         load('cadetv1p432_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p432_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==23
%         load('cadetv1p437_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p437_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==24
%         load('cadetv1p438_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p438_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==25
%         load('cadetv1p460_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p460_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==26
%         load('cadetv1p467_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p467_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    elseif a==27
%         load('cadetv1p468_entropy_drop','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
        load('cadetv1p468_entropy','filename','responsive','keep*','ori*','resp_bias','resp_uniform','spont');
    end
    
    clearvars -except a filename oris_u oris_b oris resp* oribias* oripref*
    
    % identify oris on each trial
    ori_trialu=zeros(length(oris),size(resp_uniform,2));
    ori_trialb=zeros(length(oris),size(resp_bias,2));
    for i=1:length(oris)
        [~,x]=find(oris_u==oris(i));
        [~,y]=find(oris_b==oris(i));
        Nstimu(i)=length(x);
        Nstimb(i)=length(y);
        ori_trialu(i,x)=1;
        ori_trialb(i,y)=1;
    end
    ori_trialu=logical(ori_trialu);
    ori_trialb=logical(ori_trialb);
    
    % find minimum trial # and make it the same for both
    tmp=min([Nstimu Nstimb]);
    [w,x]=min(Nstimu);
    [z,y]=min(Nstimb);
    if w==tmp
        Nstimb(y)=Nstimu(x);
    elseif z==tmp
        Nstimu(x)=Nstimb(y);
    else
        error('find minimum trial #')
    end
    
    %% GLM on all units
%     % set inputs
%     spku=resp_uniform;
%     spkb=resp_bias;
%     vsort=randperm(size(spku,1));
% %     Ns=Nstimu; spk=spku; u_oris=ori_base(1:end-1); idxs=ori_trialu; test_ori=0; vsort=1:10;
% %     30 is what I use in LDA neighbors
%     
%     for i = 1:length(oris)
%         test_ori=oris(i);
%         
%         [erroru(i,:,:,:), pred_u(i,:,:,:), proj_u(i,:,:,:), GLMori(i,:)]=TenFold_1toN_sens(Nstimu,spku,oris,test_ori,ori_trialu,vsort);
%         [errorb(i,:,:,:), pred_b(i,:,:,:), proj_b(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,oris,test_ori,ori_trialb,vsort);
%         % 9 test oris x 10 units x 8 ori pairs x 10 repeats
%         disp('next test ori')
%     end
%     stop

    %% separate OSI groups 
    
    % fixed bins of equal length
    fixed_OSI=0.01:0.33:1;
    for b=1:length(fixed_OSI)-1
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=find(oribias_u>=fixed_OSI(b) & oribias_u<=fixed_OSI(b+1));
        n_units_f=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_uniform(osi_keep,:);
        spkb=resp_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(oris)
            test_ori=oris(i);
            
            [erroru_f(i,:,:,:), pred_u_f(i,:,:,:), proj_u_f(i,:,:,:), GLMori_f(i,:)]=TenFold_1toN_sens(Nstimu,spku,oris,test_ori,ori_trialu,vsort);
            [errorb_f(i,:,:,:), pred_b_f(i,:,:,:), proj_b_f(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,oris,test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_glm_f%d',filename,b),'erroru_f','errorb_f','pred_u_f','pred_b_f','proj_u_f','proj_b_f','GLMori_f',...
            'oris','oripref*','oribias*','n_units_f','fixed_OSI')
    end
    
    % sliding bin of equal length w/ overlap
    slide_OSI=[0.1 0.4; 0.2 0.5; 0.3 0.6; 0.4 0.7; 0.5 0.8; 0.6 0.9; 0.7 1];
    for b=1:size(slide_OSI,1)
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=find(oribias_u>=slide_OSI(b,1) & oribias_u<=slide_OSI(b,2));
        n_units_s=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_uniform(osi_keep,:);
        spkb=resp_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(oris)
            test_ori=oris(i);
            
            [erroru_s(i,:,:,:), pred_u_s(i,:,:,:), proj_u_s(i,:,:,:), GLMori_s(i,:)]=TenFold_1toN_sens(Nstimu,spku,oris,test_ori,ori_trialu,vsort);
            [errorb_s(i,:,:,:), pred_b_s(i,:,:,:), proj_b_s(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,oris,test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_glm_s%d',filename,b),'erroru_s','errorb_s','pred_u_s','pred_b_s','proj_u_s','proj_b_s','GLMori_s',...
            'oris','oripref*','oribias*','n_units_s','slide_OSI')
    end
    
    % same # of units in each group but variable bin sizes
    [R,I]=sort(oribias_u);
    L=round(linspace(1,length(oribias_u),4));

    for b=1:length(L)-1
        clear error* pred* proj* GLMori spk* vsort
        osi_keep=(I(L(b):L(b+1)));
        n_units_p=length(osi_keep); % store # of units in each bin
        
        % set GLM inputs
        spku=resp_uniform(osi_keep,:);
        spkb=resp_bias(osi_keep,:);
        vsort=randperm(size(spku,1));
        
        for i = 1:length(oris)
            test_ori=oris(i);
            
            [erroru_p(i,:,:,:), pred_u_p(i,:,:,:), proj_u_p(i,:,:,:), GLMori_p(i,:)]=TenFold_1toN_sens(Nstimu,spku,oris,test_ori,ori_trialu,vsort);
            [errorb_p(i,:,:,:), pred_b_p(i,:,:,:), proj_b_p(i,:,:,:), ~]=TenFold_1toN_sens(Nstimb,spkb,oris,test_ori,ori_trialb,vsort);
            % 9 test oris x 10 units x 8 ori pars x 10 repeats
            % need to resort bc test ori gets moved to first index 1st
            % dimension each time (see GLMori)
            disp('next test ori')
        end
        save(sprintf('%s_glm_p%d',filename,b),'erroru_p','errorb_p','pred_u_p','pred_b_p','proj_u_p','proj_b_p','GLMori_p',...
            'oris','oripref*','oribias*','n_units_p','L')
    end
    
    clear ans i tmp* x y w z
%     savename=sprintf('%s_glm',filename);
%     save(savename)
    disp(filename)
end
stop
tmp=squeeze(mean(erroru(1,:,:,:),4));
tmp2=squeeze(mean(errorb(1,:,:,:),4));
figure; hold on; plot(mean(tmp,1)); plot(mean(tmp2,1))
    