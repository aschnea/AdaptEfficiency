% structure: blocks of prsentations of (i) gratings chosen from a uniform
% distribution; or (ii) a biased distribution
%   -30000 stimuli; 6 blocks of each (see 'stim')
%   -'oristim': actual orientations; 200 is the blank, 0:20:160 the gratings
%   -'base','bias': orientations divided into the two block types (15,000 each)
%   -'resp_base','resp_bias': responses to each of the 15,000
%   -'resp_base2','resp_bias2': tuning of each channel in the two blocks
%   -'resp_binned','resp_binned2': tuning curves in the uniform case, grouped by preference
%   -'resp_raw_binned','resp_raw_binned2': responses to the 15,0000 in the two grouped by preference

clear

% In the 129 files, the uniform condition (base) is screwed up for these two files
% load('129r001p174')
% load('129r001p175')
% load('130l001p170')

% 140l001p111 - blank
% 140l001p113 - awake time
% 140l001p114 - low con - bias shift
% 140l001p117 - low con
% 140l001p118 - benucci time
% 140l001p120 - oricon
% 140r001p111 - blank
% 140r001p113 - awake
% 140r001p115 - low con
% 140r001p116 - low con - bias shift
% 140r001p118 - benucci time
% 140r001p120 - oricon
% 140r001p161 - blank

% 141....p006 - awake time
% 141....p007 - awake time 6:1
% 141....p009 - awake time (fine ori)
% 141....p024 - awake time
% 141....p025 - awake time 6:1
% 141....p027 - awake time (fine ori)
% 141....p038 - awake time (fine ori) (rotated)
% 141....p039 - awake time 6:1 (rotated)
% 141....p041 - awake time (rotated)
% 141....p114 - 4:1 original
%% load nev and expo data
% % original version (100% contrast, 60ms, 15000x, blanks randomly):
data_folder=fullfile('E:\Documents\MATLAB\NEV and EXPO files');
% % 4:1 files original
% spikes=nev_reader(fullfile(data_folder,'130l001p170-s.nev'),0);
% a=ReadExpoXML('130l001#170[ensunbias3].xml');
% spikes=nev_reader(fullfile(data_folder,'140l001p108-s.nev'),0);
% a=ReadExpoXML('140l001#108[ensunbias3].xml');
% spikes=nev_reader(fullfile(data_folder,'140l001p110-s.nev'),0);
% a=ReadExpoXML('140l001#110[ensunbias3].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p107-s.nev'),0);
% a=ReadExpoXML('140r001#107[ensunbias3].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p109-s.nev'),0);
% a=ReadExpoXML('140r001#109[ensunbias3].xml');
% spikes=nev_reader('141r001p114-s.nev',0);
% a=ReadExpoXML('141r001#114[ensunbias3].xml');

% % 2:1 files original 
% spikes=nev_reader(fullfile(data_folder,'130l001p169-s.nev'),0);
% a=ReadExpoXML('130l001#169[ensunbias1].xml');
% spikes=nev_reader(fullfile(data_folder,'129r001p173-s.nev'),0);
% a=ReadExpoXML('129r001#173[ensunbias1].xml');
% spikes=nev_reader(fullfile(data_folder,'140l001p107-s.nev'),0);
% a=ReadExpoXML('140l001#107[ensunbias1].xml');
% spikes=nev_reader(fullfile(data_folder,'140l001p122-s.nev'),0);
% a=ReadExpoXML('140l001#122[ensunbias1].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p105-s.nev'),0);
% a=ReadExpoXML('140r001#105[ensunbias1].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p122-s.nev'),0);
% a=ReadExpoXML('140r001#122[ensunbias1].xml');

% % awake timing (4:1 but with 150ms - 1250x)
% spikes=nev_reader(fullfile(data_folder,'140l001p113-s.nev'),0);
% a=ReadExpoXML('140l001#113[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p113-s.nev'),0);
% a=ReadExpoXML('140r001#113[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p006-s.nev'),0);
% a=ReadExpoXML('141r001#006[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p024-s.nev'),0);
% a=ReadExpoXML('141r001#024[ensunbias3_awaketime].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p041-s.nev'),0);
% a=ReadExpoXML('141r001#041[ensunbias3_awaketime].xml');
% spikes=nev_reader('142l001p002-s.nev',0);
% a=ReadExpoXML('142l001#002[ensunbias3_awaketime].xml');
% spikes=nev_reader('142l001p007-s.nev',0);
% a=ReadExpoXML('142l001#007[ensunbias3_awaketime].xml');
% % awake timing 6:1
% spikes=nev_reader(fullfile(data_folder,'141r001p007-s.nev'),0);
% a=ReadExpoXML('141r001#007[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p025-s.nev'),0);
% a=ReadExpoXML('141r001#025[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p039-s.nev'),0);
% a=ReadExpoXML('141r001#039[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader('142l001p006-s.nev',0);
% a=ReadExpoXML('142l001#006[ensunbias3_awaketime_6to1].xml');
% spikes=nev_reader('142l001p010-s.nev',0);
% a=ReadExpoXML('142l001#010[ensunbias3_awaketime_6to1].xml');
% % awake timing fine ori (6:1)
% spikes=nev_reader(fullfile(data_folder,'141r001p009-s.nev'),0);
% a=ReadExpoXML('141r001#009[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p027-s.nev'),0);
% a=ReadExpoXML('141r001#027[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'141r001p038-s.nev'),0);
% a=ReadExpoXML('141r001#038[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'142l001p004-s.nev'),0);
% a=ReadExpoXML('142l001#004[ensunbias3_awaketime_fineori].xml');
% spikes=nev_reader(fullfile(data_folder,'142l001p009-s.nev'),0);
% a=ReadExpoXML('142l001#009[ensunbias3_awaketime_fineori].xml');

% % low contrast version (12.5% contrast, 60ms, 15000x, blanks randomly):
% spikes=nev_reader(fullfile(data_folder,'140l001p114-s.nev'),0);
% a=ReadExpoXML('140l001#114[ensunbias3_lowcon].xml');
% spikes=nev_reader(fullfile(data_folder,'140l001p117-s.nev'),0);
% a=ReadExpoXML('140l001#117[ensunbias3_lowcon].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p115-s.nev'),0);
% a=ReadExpoXML('140r001#115[ensunbias3_lowcon].xml');
% spikes=nev_reader(fullfile(data_folder,'140r001p116-s.nev'),0);
% a=ReadExpoXML('140r001#116[ensunbias3_lowcon].xml');


% % % % benucci timing     
% % % % spikes=nev_reader(fullfile(data_folder,'140l001p118-s.nev'),0); % 160 stim missing????
% % % % a=ReadExpoXML('140l001#118[ensunbias3_benucci].xml');
% % % % spikes=nev_reader(fullfile(data_folder,'140r001p118-s.nev'),0); % 160 stim missing????
% % % % a=ReadExpoXML('140r001#118[ensunbias3_benucci].xml');

% % % % % % % blank control
% % % % spikes=nev_reader(fullfile(data_folder,'140l001p111-s.nev'),0); % 160 stim missing????
% % % % a=ReadExpoXML('140l001#111[ensunbias3_blankcontrol].xml');
% % % % spikes=nev_reader(fullfile(data_folder,'140r001p111-s.nev'),0); % 160 stim missing????
% % % % a=ReadExpoXML('140r001#111[ensunbias3_blankcontrol].xml');
% % % % spikes=nev_reader(fullfile(data_folder,'140r001p161-s.nev'),0); % 160 stim missing????
% % % % a=ReadExpoXML('140r001#161[ensunbias3_blankcontrol].xml');

% % % % oricon design     % HAS DIFFERENT STATE STRUCTURE
% % % % spikes=nev_reader(fullfile(data_folder,'140l001p120-s.nev'),0);
% % % % a=ReadExpoXML('140l001#120[ensunbias3_oricon].xml');
% % % % spikes=nev_reader(fullfile(data_folder,'140r001p120-s.nev'),0);
% % % % a=ReadExpoXML('140l00r#120[ensunbias3_oricon].xml');


stim=a.passes.BlockIDs;
uniq_stim=unique(stim);
% syncs=spikes(spikes(:,1)==2000,3);
% syncs=spikes(spikes(:,1)==2000,:); % for when syncs don't match with stim
% syncs=syncs(syncs(:,2)==-3,3);
% syncs=[syncs(1)-diff(syncs(2:3));syncs]; % if missing a sync, verify than do this
%% chop stuff up
clear
for n=28:33
    clearvars -except n 
    if n==1
        load('129r001p173_preprocess_drop')
        name='129r001p173_preprocess_drop';
    elseif n==2
        load('130l001p169_preprocess_drop')
        name='130l001p169_preprocess_drop';
    elseif n==3
        load('140l001p107_preprocess_drop')
        name='140l001p107_preprocess_drop';
    elseif n==4
        load('140l001p122_preprocess_drop')
        name='140l001p122_preprocess_drop';
    elseif n==5
        load('140r001p105_preprocess_drop')
        name='140r001p105_preprocess_drop';
    elseif n==6
        load('140r001p122_preprocess_drop')
        name='140r001p122_preprocess_drop';
    elseif n==7
        load('130l001p170_preprocess_drop')
        name='130l001p170_preprocess_drop';
    elseif n==8
        load('140l001p108_preprocess_drop')
        name='140l001p108_preprocess_drop';
    elseif n==9
        load('140l001p110_preprocess_drop')
        name='140l001p110_preprocess_drop';
    elseif n==10
        load('140r001p107_preprocess_drop')
        name='140r001p107_preprocess_drop';
    elseif n==11
        load('140r001p109_preprocess_drop')
        name='140r001p109_preprocess_drop';
    elseif n==12    % begin lowcon files
        load('lowcon114_preprocess')
        name='lowcon114_preprocess';
    elseif n==13
        load('lowcon115_preprocess')
        name='lowcon115_preprocess';
    elseif n==14
        load('lowcon116_preprocess')
        name='lowcon116_preprocess';
    elseif n==15
        load('lowcon117_preprocess')
        name='lowcon117_preprocess';
    elseif n==16    % begin awaketime files
        load('140l113_awaketime_preprocess')
        name='140l113_awaketime_preprocess';
    elseif n==17
        load('140r113_awaketime_preprocess')
        name='140r113_awaketime_preprocess';
        
    elseif n==18 % start of experiment 141 files
        load('141r001p006_preprocess')
        name='141r001p006_awaketime_preprocess';
    elseif n==19
        load('141r001p007_preprocess')
        name='141r001p007_awaketime6_preprocess';
    elseif n==20
        load('141r001p009_preprocess')
        name='141r001p009_awaketime_fine_preprocess';
    elseif n==21
        load('141r001p024_preprocess')
        name='141r001p024_awaketime_preprocess';
    elseif n==22
        load('141r001p025_preprocess')
        name='141r001p025_awaketime6_preprocess';
    elseif n==23
        load('141r001p027_preprocess')
        name='141r001p027_awaketime_fine_preprocess';
    elseif n==24 % rotated fineori
        load('141r001p038_preprocess')
        name='141r001p038_awaketime_fine_preprocess';
    elseif n==25 % rotated 6:1
        load('141r001p039_preprocess')
        name='141r001p039_awaketime6_preprocess';
    elseif n==26 % rotated awaketime 4:1
        load('141r001p041_preprocess')
        name='141r001p041_awaketime_preprocess';
    elseif n==27
        load('141r001p114_preprocess')
        name='141r001p114_preprocess';
        
    elseif n==28    % start of 142 files
        load('142l001p002_preprocess')
        name='142l001p002_awaketime_preprocess';
    elseif n==29
        load('142l001p004_preprocess')
        name='142l001p004_awaketime_fine_preprocess';
    elseif n==30
        load('142l001p006_preprocess')
        name='142l001p006_awaketime6_preprocess';
    elseif n==31
        load('142l001p007_preprocess')
        name='142l001p007_awaketime_preprocess';
    elseif n==32
        load('142l001p009_preprocess')
        name='142l001p009_awaketime_fine_preprocess';
    elseif n==33
        load('142l001p010_preprocess')
        name='142l001p010_awaketime6_preprocess';
    end
    clearvars -except spikes a stim uniq_stim syncs n resp_count name
    num_units=size(resp_count,1);
    
% % % % % PULL OUT SPIKE COUNTS AT SHIFTING LATENCIES
    resp_count=zeros(num_units,length(syncs),66);
    % blks=find(diff(syncs)>0.1);
    % blks=find(stim==2);   % for awake time variant
    % spont_tmp=cell(100,1);  % for benucci and blank control
    index=1;
    for i=1:96
        resp=spikes(spikes(:,1)==i,:);
        sort_codes=unique(resp(:,2));
        sort_codes=sort_codes(sort_codes~=0 & sort_codes~=255);
%         if isempty(sort_codes)~=1
        for j=1:length(sort_codes)
            resp2=resp(resp(:,2)==sort_codes(j),3);
            % Smith's method for determing response latency (2005 paper):
            id2=1;
            for deltat=0.02:0.002:0.150 % 2ms shifts from 20-150ms
                for k=1:length(syncs)
                    resp_count(index,k,id2)=length(find(resp2>syncs(k)+deltat & resp2<=syncs(k)+deltat+diff(syncs(3:4))));
%                     if mod(k,blks(1))==0  % for Blank/Benucci variants
%                         spont_tmp{index}=[spont_tmp{index} length(find(resp2>syncs(k)+diff(syncs(3:4))...
%                             & resp2<syncs(k)+diff(syncs(3:4))+2))];
%                     end
                end
                id2=id2+1;
            end
            index=index+1;
        end
        %     end
        disp(i)
    end
    % for i=1:index-1   % for benucci and blank control
    %     spont_fr(i)=mean(spont_tmp{i}(:))/2*diff(syncs(3:4));
    %     % divide by 2 bc 2 second blank
    %     % multiply by length of grating (diff(syncs)) so time unit is same
    % end
    
% % % % % DETERMINE EXPO STIM SHOWN
    index=1;
    for i=2:length(stim)    %because first stim is blank and has no sync
        if stim(i)~=2   % 0 and 1 are bias and uniform, respectively
            temp=a.passes.events{i}.Data{end};  % surface
            ori_stim(index)=temp(2);            % ori
            temp2=a.passes.events{i}.Data{end-1};% texture
            con=temp2(4);                       % contrast
            if con==0
                ori_stim(index)=200;
            end
            index=index+1;
        else
            if isnan(ori_stim(end))~=1    % only count the blanks once since there is no sync for these
                ori_stim(index)=NaN;
                index=index+1;
            end
        end
    end
    ori_stim=ori_stim(1:end-1);     %drop last blank because it has no sync either
    ori_stim=ori_stim(isnan(ori_stim)~=1);
    ori_stim(ori_stim==180)=0;      % for 2:1 files
    
% % % % % SEPARATE UNIFORM AND BIASED STIM AND RESPONSES
    base=[];        % ori sequence in uniform
    bias=[];        % ori sequence in bias
    temp_base=[];
    temp_bias=[];
    if length(stim)>20000 && length(stim)<30000
        last=20;
        ori_base=[(0:10:170) 200]; % unique stim
    else
        last=12;
        ori_base=[(0:20:160) 200]; % unique stim
    end
    for i=1:last % original/lowcon/blank = 12x2500; awake/6:1 = 12x1250; fineori=20x1250; benucci=20x4900; oricon=20x1250
        if mod(i,2)~=0
%             base=[base ori_stim((i-1)*2500+400:i*2500)];     % drop first 400 trials (~28s) after each switch
%             temp_base=[temp_base (i-1)*2500+400:i*2500];
%             base=[base ori_stim((i-1)*2500+1:i*2500)];     %ORIGINAL/LOWCON/BLANK
%             temp_base=[temp_base (i-1)*2500+1:i*2500];
            base=[base ori_stim((i-1)*1250+1:i*1250)];     %AWAKE/ORICON
            temp_base=[temp_base (i-1)*1250+1:i*1250];
    %         base=[base ori_stim((i-1)*4900+1:i*4900)];     %BENUCCI
    %         temp_base=[temp_base (i-1)*4900+1:i*4900];
        else
%             bias=[bias ori_stim((i-1)*2500+400:i*2500)];     % drop first 400 trials (~28s) after each switch
%             temp_bias=[temp_bias (i-1)*2500+400:i*2500];
%             bias=[bias ori_stim((i-1)*2500+1:i*2500)];     %ORIGINAL/LOWCON/BLANK
%             temp_bias=[temp_bias (i-1)*2500+1:i*2500];
            bias=[bias ori_stim((i-1)*1250+1:i*1250)];     %AWAKE/ORICON
            temp_bias=[temp_bias (i-1)*1250+1:i*1250];
    %         bias=[bias ori_stim((i-1)*4900+1:i*4900)];     %BENUCCI
    %         temp_bias=[temp_bias (i-1)*4900+1:i*4900];
        end
    end
    
    resp_raw_base_test=zeros(size(resp_count,1),66,size(base,2));   % spike counts in uniform
    resp_raw_bias_test=zeros(size(resp_count,1),66,size(bias,2));   % spike counts in bias
    for j=1:size(resp_count,1)
        for k=1:66
            resp_raw_base_test(j,k,:)=resp_count(j,temp_base,k);    
            resp_raw_bias_test(j,k,:)=resp_count(j,temp_bias,k);
        end
    end
% % % % % CALCULATE TUNING CURVES (AVG RESPONSES) &
% % % % % DETERMINE BEST LATENCY TO USE (KEEP THAT TC AND SPIKE COUNT)
    for j=1:size(resp_count,1)
        tmp=zeros(1,66);
        for k = 1:66
            for i=1:length(ori_base)
                temp=find(base==ori_base(i));
                resp_base_test(j,k,i)=mean(resp_raw_base_test(j,k,temp));
                temp=find(bias==ori_base(i));
                resp_bias_test(j,k,i)=mean(resp_raw_bias_test(j,k,temp));
            end
            tmp(k)=var(squeeze(resp_base_test(j,k,1:9)));
        end
        [~,y]=find(tmp==max(tmp));
        if length(y)>1
            y=y(1);
        end
        disp(y)
        if isempty(y)
            resp_raw_base(j,:)=nan;
            resp_raw_bias(j,:)=nan;
            resp_base(j,:)=nan;
            resp_bias(j,:)=nan;
            latency(j)=nan;
        else
            resp_raw_base(j,:)=squeeze(resp_raw_base_test(j,y,:));
            resp_raw_bias(j,:)=squeeze(resp_raw_bias_test(j,y,:));
            resp_base(j,:)=squeeze(resp_base_test(j,y,:));
            resp_bias(j,:)=squeeze(resp_bias_test(j,y,:));
            latency(j)=y*0.002+0.02;
        end
    end
    % resp_base(:,10)=spont_fr;
    % resp_bias(:,10)=spont_fr;

% % % % % DETERMINE ORIENTATION PREFERENCE OF EACH UNIT USING BEST LATENCY
    for i=1:size(resp_base,1)
        [~,~,oribias(i),oripref(i),~,~] = orivecfit(ori_base(1:end-1),resp_base(i,1:end-1),resp_base(i,end)); 
        % for benucci and blank version use:
    %     [~,~,oribias(i),oripref(i),~,~] = orivecfit((0:20:160),resp_base(i,1:9),spont_fr(i));
    end
    % this makes 0 preferring cells fire most to a 0 stimulus in EXPO
    oripref(oripref>180)=oripref(oripref>180)-180;

    clear i e id2 index j k tmp* temp* resp resp2 deltat con...
        resp_raw_base_test resp_raw_bias_test
    save(name);
end

akstop
% save as *_preprocess

figure
for j=1:66
    hold on
    plot(squeeze(resp_base_test(40,j,:)),'k')
    plot(squeeze(resp_bias_test(40,j,:)),'r')
    title(j)
    axis square
    pause; clf
end

stop
figure
for j=1:size(resp_base,1)
    hold on
    plot(resp_base(j,1:end-1),'k')
    plot(resp_bias(j,1:end-1),'r')
    title([j oripref(j) oribias(j)])
    pause;clf
end
keep=[1:5 7 9:19 21:28 30:34 36 38:45];
latency=latency(keep);
num_units=length(keep);
oribias=oribias(keep);
oripref=oripref(keep);
resp_base=resp_base(keep,:);
resp_base_test=resp_base_test(keep,:,:);
resp_bias=resp_bias(keep,:);
resp_bias_test=resp_bias_test(keep,:,:);
resp_count=resp_count(keep,:,:);
resp_raw_base=resp_raw_base(keep,:);
resp_raw_bias=resp_raw_bias(keep,:);
%% load preprocessed

clear
% original:   % 4x bias
% load('130l001p170_preprocess')
% load('140l001p108_preprocess')
% load('140l001p110_preprocess')
% load('140r001p107_preprocess')
% load('140r001p109_preprocess')

% low conrast:
% load('lowcon114_preprocess')
% load('lowcon115_preprocess')
% load('lowcon116_preprocess')
% load('lowcon117_preprocess')

% awake time:
% load('140l113_awaketime_preprocess')
% load('140r113_awaketime_preprocess')

% no 160 stim in bias??????
% blank control
% load('140l111_blank_preprocess')
% load('140r111_blank_preprocess')
% load('140r161_blank_preprocess')

% Benucci time:
% load('140l118_benucci_preprocess')
% load('140r118_benucci_preprocess')
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
% resp_base(:,10)=spont_fr;
% resp_bias(:,10)=spont_fr;
% group neurons by their ori pref
for i=1:size(resp_base,1)
    [~,~,oribias(i),oripref(i),~,~] = orivecfit((0:20:160),resp_base(i,1:9),resp_base(i,10)); 
    % for benucci and blank version use:
%     [~,~,oribias(i),oripref(i),~,~] = orivecfit((0:20:160),resp_base(i,1:9),spont_fr(i));
end
oripref(oripref>180)=oripref(oripref>180)-180;
%this makes 0 preferring cells fire most to a 0 stimulus in EXPO

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
    pred_resp_base(:,j) =resp_base_binned(:,find(ori_base==base(j)));
    pred_resp_biasx(:,j)=resp_base_binned(:,find(ori_base==bias(j)));
    pred_resp_bias(:,j) =resp_bias_binned(:,find(ori_base==bias(j)));
end

%count number of each stimuli
for i=1:length(ori_base)
    base_count(i)=length(find(base==ori_base(i)));
    bias_count(i)=length(find(bias==ori_base(i))); %bias redefined to have 180=0 above
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

% first 200 trials = 14s
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
        if aa==1
            model_pred_base(j,i)=nl_out_base(i,aa);
        else
            model_pred_base(j,i)=nanmean(nl_out_base(i,aa-1:aa));
        end
        
        aa=find(nl_in_bias(i,:)>filt_out_basex(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        if aa==1
            model_pred_basex(j,i)=nl_out_bias(i,aa);
        else
            model_pred_basex(j,i)=nanmean(nl_out_bias(i,aa-1:aa));
        end
        
        aa=find(nl_in_bias(i,:)>filt_out_bias(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        if aa==1
            model_pred_bias(j,i)=nl_out_bias(i,aa);
        else
            model_pred_bias(j,i)=nanmean(nl_out_bias(i,aa-1:aa));
        end
        
        aa=find(nl_in_base(i,:)>filt_out_biasx(j,i));
        if isempty(aa)
            aa=size(nl_in_base,2);
        else
            aa=aa(1);
        end
        if aa==1
            model_pred_biasx(j,i)=nl_out_base(i,aa);
        else
            model_pred_biasx(j,i)=nanmean(nl_out_base(i,aa-1:aa));
        end            
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
bar((0:20:180),[temp base_count(end)]/length(base),1);
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
axis([-10 180 0.7 1.3])
plot((0:20:160),circshift(nanmean(model_pred_base_norm,2),[4 0]),'b','LineWidth',2)
plot((0:20:160),circshift(nanmean(model_pred_basex_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),ones(9,1),':r')

subplot(2,2,3)
temp=circshift(bias_count(1:end-1),[0 4]);
bar((0:20:180),[temp bias_count(end)]/length(base),1);
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
axis([-10 180 0.7 1.3])
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
plot((0:20:160),circshift(mean(pred_resp_biasx_norm,2),[4 0]),'r')

%% simpler plot--data

figure

subplot(2,2,1)
temp=circshift(base_count(1:end-1),[0 4]);
bar((0:20:180),[temp base_count(end)]/length(base),1);
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
axis([-10 180 0.8 1.3])
plot((0:20:160),circshift(mean(pred_resp_basex_norm,2),[4 0]),'r','LineWidth',2)
plot((0:20:160),ones(9,1),':r')

subplot(2,2,3)
temp=circshift(bias_count(1:end-1),[0 4]);
bar((0:20:180),[temp bias_count(end)]/length(base),1);
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
axis([-10 180 0.8 1.3])
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


%% plots

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