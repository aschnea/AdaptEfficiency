%% Tuning curve model:
clear
%% basic model setup
trials=3100;  % for awake sessions: avg=3130 median=3057    % was 10000
% oris=10:20:170;
oris=10:20:170;       % variation
neurons=70;         % 100 864    %537;   % this is close to # of neurons in data and allows for integer ori prefs
ori_prefs=0:180/neurons:180;    %1:179/num_neurons:180-179/num_neurons;
adapt_strength=(0.1:0.2:0.5);   % awake kernel looks like ~10%

% r_ttl=nan*ones(num_neurons*(num_neurons-1)/2,6);
corr_u_ak=nan*ones(70,70);
corr_b_ak=nan*ones(70,70);
corr_ua_ak=nan*ones(70,70);
corr_ba_ak=nan*ones(70,70);
corr_u_ak2=nan*ones(70,70);
corr_b_ak2=nan*ones(70,70);
corr_ua_ak2=nan*ones(70,70);
corr_ba_ak2=nan*ones(70,70);
corr_u_ak_w=nan*ones(70,70);
corr_b_ak_w=nan*ones(70,70);
corr_ua_ak_w=nan*ones(70,70);
corr_ba_ak_w=nan*ones(70,70);
corr_u_ak_w2=nan*ones(70,70);
corr_b_ak_w2=nan*ones(70,70);
corr_ua_ak_w2=nan*ones(70,70);
corr_ba_ak_w2=nan*ones(70,70);

for rr=1:length(adapt_strength)
    % adaptation kernel (for adapted curves)
%     rr=length(adapt_strength);   % strongest kernel
%     kernel=exp(6*(cos(2*deg2rad(oris-90))-1)); % kernel centered at 90 original
%     kernel=1-adapt_strength(rr)*kernel;
    kernelw=0.1+exp(2*(cos(2*deg2rad(oris-90))-1)); % kernel centered at 90
    kernelw=1-adapt_strength(rr)*kernelw;
    kerneln=0.1+exp(10*(cos(2*deg2rad(oris-90))-1)); % kernel centered at 90
    kerneln=1-adapt_strength(rr)*kerneln;
    for qq=1:length(adapt_strength)     % # of kernels
        % create tuning curves: amir's way
%         k=3;                        % sets tuning curve width; gives half width of ~24 deg
%         tmp=2*(rand(neurons,1)-0.5);% gives numbers -1 to 1
%         k=k+3.9*tmp;                % bandwidth
%         ampl=10*(rand(neurons,1)+0.1); % amplitude
%         offset=5*(0.2+rand(neurons,1));
%         for i=1:neurons                % defines the tuning of each cell
%             tune(i,:)=ampl(i)*exp(k(i)*(cos((pi*(oris-ori_prefs(i))/90))-1))+offset(i);
%         end
%         tune_a=tune.*kernel;

        % create tuning curves: adam's way
        ampl=11*rand(neurons,1);
        base=1*rand(neurons,1);
        % wide bandwidth:
        tmp=rand(neurons,1)-0.5;
        bww=2+5*tmp; % originally 3.5 + 8
        bww(bww<0)=0.01;
        % add a second, narrower bw
        bwn=8+10*tmp;
        bwn(bwn<0)=0.01;
        for i=1:neurons
            % wide bw:
            aktune_ww(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(2*deg2rad(oris-ori_prefs(i)))-1));
            aktune_a_ww(i,:)=aktune_ww(i,:).*kernelw;
            aktune_wn(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(2*deg2rad(oris-ori_prefs(i)))-1));
            aktune_a_wn(i,:)=aktune_wn(i,:).*kerneln;
            % narrow bw:
            aktune_nw(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(2*deg2rad(oris-ori_prefs(i)))-1));
            aktune_a_nw(i,:)=aktune_nw(i,:).*kernelw;
            aktune_nn(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(2*deg2rad(oris-ori_prefs(i)))-1));
            aktune_a_nn(i,:)=aktune_nn(i,:).*kerneln;
        end
        if qq==3
            figure;
            subplot(231);
            plot(circshift(aktune_ww(26:5:46,:)',0),'k');
            axis square; box off
            title('Unadapted tuning curves (ww)')
            set(gca,'TickDir','out')
            ylabel('Wide BW: Response')

            subplot(232); hold on
            plot((kernelw),'r');
            axis square; box off
            title('Adaptation kernel (w)')
            set(gca,'Xtick',[0 30 60 90 120 150 180],'xticklabel',{'-90','-60','-30','0','30','60','90'},'TickDir','out')
            ylim([0 1]);

            subplot(233); hold on
            plot(circshift(aktune_ww(26:5:46,:)',0),'k');
            plot(circshift(aktune_a_ww(26:5:46,:)',0),'r');
            axis square; box off
            title('Adapted tuning curves (ww)')
            set(gca,'TickDir','out')

            subplot(234);
            plot(circshift(aktune_nw(26:5:46,:)',0),'k');
            axis square; box off
            title('Unadapted tuning curves (nw)')
            set(gca,'TickDir','out')
            ylabel('Narrow BW: Response')

            subplot(235); hold on
            plot((kernelw),'r');
            axis square; box off
            title('Adaptation kernel (w)')
            set(gca,'Xtick',[0 30 60 90 120 150 180],'xticklabel',{'-90','-60','-30','0','30','60','90'},'TickDir','out')
            ylim([0 1]);

            subplot(236); hold on
            plot(circshift(aktune_nw(26:5:46,:)',0),'k');
            plot(circshift(aktune_a_nw(26:5:46,:)',0),'r');
            axis square; box off
            title('Adapted tuning curves (nw)')
            set(gca,'TickDir','out')

        end
%         stop
        %% vanilla model - no rate adapt, no normalization
        % generate stimulus distributions using 6:1 ratio:
        uniform_dist=randi(9,trials,1);
        biased_dist=randi(14,trials,1);
        biased_dist(biased_dist>9)=5;   % bias == 90
%         uniform_dist=randi(71,trials,1);
%         biased_dist=randi(77,trials,1);
%         biased_dist(biased_dist>71)=37;

        % generate responses to distributions:
        for i=1:neurons     
            % log10 was for when we cared about divergence. Don't need anymore
%             resp_u_nw(i,:)=log10(poissrnd(aktune_nw(i,uniform_dist))+1);
%             resp_b_nw(i,:)=log10(poissrnd(aktune_nw(i,biased_dist))+1);
%             resp_ua_nw(i,:)=log10(poissrnd(aktune_a_nw(i,uniform_dist))+1);
%             resp_ba_nw(i,:)=log10(poissrnd(aktune_a_nw(i,biased_dist))+1);
%             resp_u_ww(i,:)=log10(poissrnd(aktune_ww(i,uniform_dist))+1);
%             resp_b_ww(i,:)=log10(poissrnd(aktune_ww(i,biased_dist))+1);
%             resp_ua_ww(i,:)=log10(poissrnd(aktune_a_ww(i,uniform_dist))+1);
%             resp_ba_ww(i,:)=log10(poissrnd(aktune_a_ww(i,biased_dist))+1);
%             
%             resp_u_nn(i,:)=log10(poissrnd(aktune_nn(i,uniform_dist))+1);
%             resp_b_nn(i,:)=log10(poissrnd(aktune_nn(i,biased_dist))+1);
%             resp_ua_nn(i,:)=log10(poissrnd(aktune_a_nn(i,uniform_dist))+1);
%             resp_ba_nn(i,:)=log10(poissrnd(aktune_a_nn(i,biased_dist))+1);
%             resp_u_wn(i,:)=log10(poissrnd(aktune_wn(i,uniform_dist))+1);
%             resp_b_wn(i,:)=log10(poissrnd(aktune_wn(i,biased_dist))+1);
%             resp_ua_wn(i,:)=log10(poissrnd(aktune_a_wn(i,uniform_dist))+1);
%             resp_ba_wn(i,:)=log10(poissrnd(aktune_a_wn(i,biased_dist))+1);
            
            resp_u_nw(i,:)=poissrnd(aktune_nw(i,uniform_dist));
            resp_b_nw(i,:)=poissrnd(aktune_nw(i,biased_dist));
            resp_ua_nw(i,:)=poissrnd(aktune_a_nw(i,uniform_dist));
            resp_ba_nw(i,:)=poissrnd(aktune_a_nw(i,biased_dist));
            resp_u_ww(i,:)=poissrnd(aktune_ww(i,uniform_dist));
            resp_b_ww(i,:)=poissrnd(aktune_ww(i,biased_dist));
            resp_ua_ww(i,:)=poissrnd(aktune_a_ww(i,uniform_dist));
            resp_ba_ww(i,:)=poissrnd(aktune_a_ww(i,biased_dist));
            
            resp_u_nn(i,:)=poissrnd(aktune_nn(i,uniform_dist));
            resp_b_nn(i,:)=poissrnd(aktune_nn(i,biased_dist));
            resp_ua_nn(i,:)=poissrnd(aktune_a_nn(i,uniform_dist));
            resp_ba_nn(i,:)=poissrnd(aktune_a_nn(i,biased_dist));
            resp_u_wn(i,:)=poissrnd(aktune_wn(i,uniform_dist));
            resp_b_wn(i,:)=poissrnd(aktune_wn(i,biased_dist));
            resp_ua_wn(i,:)=poissrnd(aktune_a_wn(i,uniform_dist));
            resp_ba_wn(i,:)=poissrnd(aktune_a_wn(i,biased_dist));
        end
        
        % calculate correlations:
        for nn=1:neurons-1
            for mm=nn+1:neurons                
                corr_u_ak(nn,mm)=corr(resp_u_nw(nn,:)',resp_u_nw(mm,:)');
                corr_b_ak(nn,mm)=corr(resp_b_nw(nn,:)',resp_b_nw(mm,:)');
                corr_ua_ak(nn,mm)=corr(resp_ua_nw(nn,:)',resp_ua_nw(mm,:)');
                corr_ba_ak(nn,mm)=corr(resp_ba_nw(nn,:)',resp_ba_nw(mm,:)');
                corr_u_ak_w(nn,mm)=corr(resp_u_ww(nn,:)',resp_u_ww(mm,:)');
                corr_b_ak_w(nn,mm)=corr(resp_b_ww(nn,:)',resp_b_ww(mm,:)');
                corr_ua_ak_w(nn,mm)=corr(resp_ua_ww(nn,:)',resp_ua_ww(mm,:)');
                corr_ba_ak_w(nn,mm)=corr(resp_ba_ww(nn,:)',resp_ba_ww(mm,:)');
                
                corr_u_ak2(nn,mm)=corr(resp_u_nn(nn,:)',resp_u_nn(mm,:)');
                corr_b_ak2(nn,mm)=corr(resp_b_nn(nn,:)',resp_b_nn(mm,:)');
                corr_ua_ak2(nn,mm)=corr(resp_ua_nn(nn,:)',resp_ua_nn(mm,:)');
                corr_ba_ak2(nn,mm)=corr(resp_ba_nn(nn,:)',resp_ba_nn(mm,:)');
                corr_u_ak_w2(nn,mm)=corr(resp_u_wn(nn,:)',resp_u_wn(mm,:)');
                corr_b_ak_w2(nn,mm)=corr(resp_b_wn(nn,:)',resp_b_wn(mm,:)');
                corr_ua_ak_w2(nn,mm)=corr(resp_ua_wn(nn,:)',resp_ua_wn(mm,:)');
                corr_ba_ak_w2(nn,mm)=corr(resp_ba_wn(nn,:)',resp_ba_wn(mm,:)');
            end
        end
        % reflect for full matrix:
        for e=1:length(corr_u_ak)
            for i =1:length(corr_u_ak)
                if e>i
                    corr_u_ak(e,i)=corr_u_ak(i,e);
                    corr_b_ak(e,i)=corr_b_ak(i,e);
                    corr_ua_ak(e,i)=corr_ua_ak(i,e);
                    corr_ba_ak(e,i)=corr_ba_ak(i,e);
                    corr_u_ak_w(e,i)=corr_u_ak_w(i,e);
                    corr_b_ak_w(e,i)=corr_b_ak_w(i,e);
                    corr_ua_ak_w(e,i)=corr_ua_ak_w(i,e);
                    corr_ba_ak_w(e,i)=corr_ba_ak_w(i,e);
                    
                    corr_u_ak2(e,i)=corr_u_ak2(i,e);
                    corr_b_ak2(e,i)=corr_b_ak2(i,e);
                    corr_ua_ak2(e,i)=corr_ua_ak2(i,e);
                    corr_ba_ak2(e,i)=corr_ba_ak2(i,e);
                    corr_u_ak_w2(e,i)=corr_u_ak_w2(i,e);
                    corr_b_ak_w2(e,i)=corr_b_ak_w2(i,e);
                    corr_ua_ak_w2(e,i)=corr_ua_ak_w2(i,e);
                    corr_ba_ak_w2(e,i)=corr_ba_ak_w2(i,e);
                end
            end
        end
        
        CORR_U_ak(qq,rr,:,:)=corr_u_ak;
        CORR_B_ak(qq,rr,:,:)=corr_b_ak;
        CORR_UA_ak(qq,rr,:,:)=corr_ua_ak;
        CORR_BA_ak(qq,rr,:,:)=corr_ba_ak;
        CORR_U_ak_w(qq,rr,:,:)=corr_u_ak_w;
        CORR_B_ak_w(qq,rr,:,:)=corr_b_ak_w;
        CORR_UA_ak_w(qq,rr,:,:)=corr_ua_ak_w;
        CORR_BA_ak_w(qq,rr,:,:)=corr_ba_ak_w;
        
        CORR_U_ak2(qq,rr,:,:)=corr_u_ak2;
        CORR_B_ak2(qq,rr,:,:)=corr_b_ak2;
        CORR_UA_ak2(qq,rr,:,:)=corr_ua_ak2;
        CORR_BA_ak2(qq,rr,:,:)=corr_ba_ak2;
        CORR_U_ak_w2(qq,rr,:,:)=corr_u_ak_w2;
        CORR_B_ak_w2(qq,rr,:,:)=corr_b_ak_w2;
        CORR_UA_ak_w2(qq,rr,:,:)=corr_ua_ak_w2;
        CORR_BA_ak_w2(qq,rr,:,:)=corr_ba_ak_w2;
        
        disp([rr qq])
    end
end
clear e i nn mm tmp rr qq k 

%% correlations plots
oris_fine=0:1:180;
for rr=1:length(adapt_strength)
    adapt_kernelw=exp(2*(cos(2*deg2rad(oris_fine-90))-1));
    adapt_kernelw=1-adapt_strength(rr)*adapt_kernelw;
    ADAPT_KERNELw(rr,:)=adapt_kernelw;
    adapt_kerneln=exp(8*(cos(2*deg2rad(oris_fine-90))-1));
    adapt_kerneln=1-adapt_strength(rr)*adapt_kerneln;
    ADAPT_KERNELn(rr,:)=adapt_kerneln;
end

aaak=squeeze(mean(CORR_U_ak,1)); %aaak=circshift(aaak,50,1); %aaak=circshift(aaak,50,2);
bbak=squeeze(mean(CORR_B_ak,1)); %bbak=circshift(bbak,50,1); %bbak=circshift(bbak,50,2);
ccak=squeeze(mean(CORR_UA_ak,1)); %ccak=circshift(ccak,50,1); %ccak=circshift(ccak,50,2);
ddak=squeeze(mean(CORR_BA_ak,1)); %ddak=circshift(ddak,50,1); %ddak=circshift(ddak,50,2);
aaak_w=squeeze(mean(CORR_U_ak_w,1)); %aaak_w=circshift(aaak_w,50,1); %aaak_w=circshift(aaak_w,50,2);
bbak_w=squeeze(mean(CORR_B_ak_w,1)); %bbak_w=circshift(bbak_w,50,1); %bbak_w=circshift(bbak_w,50,2);
ccak_w=squeeze(mean(CORR_UA_ak_w,1));%ccak_w=circshift(ccak_w,50,1); %ccak_w=circshift(ccak_w,50,2);
ddak_w=squeeze(mean(CORR_BA_ak_w,1));%ddak_w=circshift(ddak_w,50,1); %ddak_w=circshift(ddak_w,50,2);

figure
sub_index=0;
for rr=1:2:size(ADAPT_KERNELw,1)
    subplot(2,6,sub_index*6+1)
    imagesc(squeeze(aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Uniform,unadapted')
    end
    subplot(2,6,sub_index*6+2)
    imagesc(squeeze(bbak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, unadapted resp')
    end
    subplot(2,6,sub_index*6+3)
    imagesc(squeeze(bbak(rr,:,:)-aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Diff-the problem')
    end
    subplot(2,6,sub_index*6+4)
    imagesc(squeeze(ddak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, adapted resp')
    end
    subplot(2,6,sub_index*6+5)
    imagesc(squeeze(ddak(rr,:,:)-aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Difference with uniform')
    end
    subplot(2,6,sub_index*6+6)
    plot(ADAPT_KERNELw(rr,:),'r')
    axis([0 180 0 1])
    axis square;box off
    
    sub_index=sub_index+1;
end
supertitle('narrow bw, wide kernel')
figure
sub_index=0;
for rr=1:2:size(ADAPT_KERNELw,1)
    subplot(2,6,sub_index*6+1)
    imagesc(squeeze(aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Uniform,unadapted')
    end
    subplot(2,6,sub_index*6+2)
    imagesc(squeeze(bbak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, unadapted resp')
    end
    subplot(2,6,sub_index*6+3)
    imagesc(squeeze(bbak_w(rr,:,:)-aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Diff-the problem')
    end
    subplot(2,6,sub_index*6+4)
    imagesc(squeeze(ddak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, adapted resp')
    end
    subplot(2,6,sub_index*6+5)
    imagesc(squeeze(ddak_w(rr,:,:)-aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Difference with uniform')
    end
    subplot(2,6,sub_index*6+6)
    plot(ADAPT_KERNELw(rr,:),'r')
    axis([0 180 0 1])
    axis square;box off
    
    sub_index=sub_index+1;
end
supertitle('wide bw, wide kernel')

aaak=squeeze(mean(CORR_U_ak2,1));
bbak=squeeze(mean(CORR_B_ak2,1));
ccak=squeeze(mean(CORR_UA_ak2,1));
ddak=squeeze(mean(CORR_BA_ak2,1));
aaak_w=squeeze(mean(CORR_U_ak_w2,1));
bbak_w=squeeze(mean(CORR_B_ak_w2,1));
ccak_w=squeeze(mean(CORR_UA_ak_w2,1));
ddak_w=squeeze(mean(CORR_BA_ak_w2,1));

figure
sub_index=0;
for rr=1:2:size(ADAPT_KERNELn,1)
    subplot(2,6,sub_index*6+1)
    imagesc(squeeze(aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Uniform,unadapted')
    end
    subplot(2,6,sub_index*6+2)
    imagesc(squeeze(bbak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, unadapted resp')
    end
    subplot(2,6,sub_index*6+3)
    imagesc(squeeze(bbak(rr,:,:)-aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Diff-the problem')
    end
    subplot(2,6,sub_index*6+4)
    imagesc(squeeze(ddak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, adapted resp')
    end
    subplot(2,6,sub_index*6+5)
    imagesc(squeeze(ddak(rr,:,:)-aaak(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Difference with uniform')
    end
    subplot(2,6,sub_index*6+6)
    plot(ADAPT_KERNELw(rr,:),'r')
    axis([0 180 0 1])
    axis square;box off
    
    sub_index=sub_index+1;
end
supertitle('narrow bw, narrow kernel')
figure
sub_index=0;
for rr=1:2:size(ADAPT_KERNELn,1)
    subplot(2,6,sub_index*6+1)
    imagesc(squeeze(aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Uniform,unadapted')
    end
    subplot(2,6,sub_index*6+2)
    imagesc(squeeze(bbak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, unadapted resp')
    end
    subplot(2,6,sub_index*6+3)
    imagesc(squeeze(bbak_w(rr,:,:)-aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Diff-the problem')
    end
    subplot(2,6,sub_index*6+4)
    imagesc(squeeze(ddak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Biased dist, adapted resp')
    end
    subplot(2,6,sub_index*6+5)
    imagesc(squeeze(ddak_w(rr,:,:)-aaak_w(rr,:,:)),[-1 1])
    axis square;box off
    if rr==1
        title('Difference with uniform')
    end
    subplot(2,6,sub_index*6+6)
    plot(ADAPT_KERNELw(rr,:),'r')
    axis([0 180 0 1])
    axis square;box off
    
    sub_index=sub_index+1;
end
supertitle('wide bw, narrow kernel')

save('TCmodel_2kern_corr')
stop
%% linear discriminant analysis
clear

% set trials, pop size, and ori prefs/oris used
trials=3000;        % approximate avg of trials in awake data
neurons=70;         % approximate avg of # of units in awake data
trial_match=300;    % approximate avg of trials used in neighbors classification in awake data
oris_coarse=10:20:170;   % oris used in experiment
oris_fine=0:2.5:177.5;  % oris for neighbors classification
% oris_fine=0:5:175;  % oris for neighbors classification
ori_prefs=0:180/neurons:180;

% create tuning curves: adam's way
ampl=11*ones(neurons);%*rand(neurons,1);
base=1*ones(neurons);%*rand(neurons,1);
% wide bandwidth:
tmp=rand(neurons,1)-0.5;
bww=2*ones(neurons);%+5*tmp; % originally 3.5 + 8
bww(bww<0)=0.01;
% add a second, narrower bw
bwn=8*ones(neurons);%+10*tmp;
bwn(bwn<0)=0.01;

% wide kernel:  using strong (0.5) kernel ----- was (cos(2*deg2rad...)
kernelw=1-0.5*exp(3*(cos(deg2rad(oris_coarse-90))-1)); % kernel centered at 90
% narrow kernel:
kerneln=1-0.5*exp(10*(cos(deg2rad(oris_coarse-90))-1)); % kernel centered at 90

for i=1:neurons
    % wide bw:
    aktune_ww(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_ww(i,:)=aktune_ww(i,:).*kernelw;
    aktune_wn(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_wn(i,:)=aktune_wn(i,:).*kerneln;
    % narrow bw:
    aktune_nw(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_nw(i,:)=aktune_nw(i,:).*kernelw;
    aktune_nn(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_nn(i,:)=aktune_nn(i,:).*kerneln;
end
% divide by 14 to approximate FR of 70ms epoch
% aktune_ww=aktune_ww./14;
% aktune_a_ww=aktune_a_ww./14;
% aktune_wn=aktune_wn./14;
% aktune_a_wn=aktune_a_wn./14;
% aktune_nw=aktune_nw./14;
% aktune_a_nw=aktune_a_nw./14;
% aktune_nn=aktune_nn./14;
% aktune_a_nn=aktune_a_nn./14;

u_dist=randi(9,trials,1);
b_dist=randi(14,trials,1);
b_dist(b_dist>9)=5;   % bias == 90

for i=1:neurons            
    resp_u_nw(i,:)=poissrnd(aktune_nw(i,u_dist));       % UTC trained on Ustats
    resp_b_nw(i,:)=poissrnd(aktune_nw(i,b_dist));       % UTC trained on Bstats
    resp_ua_nw(i,:)=poissrnd(aktune_a_nw(i,u_dist));    % BTC trained on Ustats
    resp_ba_nw(i,:)=poissrnd(aktune_a_nw(i,b_dist));    % BTC trained on Bstats
    resp_u_ww(i,:)=poissrnd(aktune_ww(i,u_dist));
    resp_b_ww(i,:)=poissrnd(aktune_ww(i,b_dist));
    resp_ua_ww(i,:)=poissrnd(aktune_a_ww(i,u_dist));
    resp_ba_ww(i,:)=poissrnd(aktune_a_ww(i,b_dist));
    
    resp_u_nn(i,:)=poissrnd(aktune_nn(i,u_dist));
    resp_b_nn(i,:)=poissrnd(aktune_nn(i,b_dist));
    resp_ua_nn(i,:)=poissrnd(aktune_a_nn(i,u_dist));
    resp_ba_nn(i,:)=poissrnd(aktune_a_nn(i,b_dist));
    resp_u_wn(i,:)=poissrnd(aktune_wn(i,u_dist));
    resp_b_wn(i,:)=poissrnd(aktune_wn(i,b_dist));
    resp_ua_wn(i,:)=poissrnd(aktune_a_wn(i,u_dist));
    resp_ba_wn(i,:)=poissrnd(aktune_a_wn(i,b_dist));
    
%     resp_u_nw(i,:)=round(aktune_nw(i,u_dist));       % UTC trained Ustats
%     resp_b_nw(i,:)=round(aktune_nw(i,b_dist));       % UTC trained Bstats
%     resp_ua_nw(i,:)=round(aktune_a_nw(i,u_dist));    % BTC trained Ustats
%     resp_ba_nw(i,:)=round(aktune_a_nw(i,b_dist));    % BTC trained Bstats
%     resp_u_ww(i,:)=round(aktune_ww(i,u_dist));
%     resp_b_ww(i,:)=round(aktune_ww(i,b_dist));
%     resp_ua_ww(i,:)=round(aktune_a_ww(i,u_dist));
%     resp_ba_ww(i,:)=round(aktune_a_ww(i,b_dist));
%     
%     resp_u_nn(i,:)=round(aktune_nn(i,u_dist));
%     resp_b_nn(i,:)=round(aktune_nn(i,b_dist));
%     resp_ua_nn(i,:)=round(aktune_a_nn(i,u_dist));
%     resp_ba_nn(i,:)=round(aktune_a_nn(i,b_dist));
%     resp_u_wn(i,:)=round(aktune_wn(i,u_dist));
%     resp_b_wn(i,:)=round(aktune_wn(i,b_dist));
%     resp_ua_wn(i,:)=round(aktune_a_wn(i,u_dist));
%     resp_ba_wn(i,:)=round(aktune_a_wn(i,b_dist));
end
train=0.9*trials;

% prediction = classify(test,train,stims, 'linear');
% [lda_u, lda_err_u]=classify(resp_u_nw(:,train+1:end)',resp_u_nw(:,1:train)',u_dist(1:train));
% [lda_b, lda_err_b]=classify(resp_ba_nw(:,train+1:end)',resp_ba_nw(:,1:train)',b_dist(1:train));
% [lda_up, lda_err_up]=classify(resp_ua_nw(:,train+1:end)',resp_ua_nw(:,1:train)',u_dist(1:train));
% [lda_bp, lda_err_bp]=classify(resp_b_nw(:,train+1:end)',resp_b_nw(:,1:train)',b_dist(1:train));
[lda_u, lda_err_u]=classify(resp_u_nw(:,train+1:end)',resp_u_nw(:,1:train)',u_dist(1:train));
[lda_b, lda_err_b]=classify(resp_ba_nw(:,train+1:end)',resp_ua_nw(:,1:train)',u_dist(1:train));
[lda_up, lda_err_up]=classify(resp_ua_nw(:,train+1:end)',resp_ua_nw(:,1:train)',u_dist(1:train));
[lda_bp, lda_err_bp]=classify(resp_b_nw(:,train+1:end)',resp_u_nw(:,1:train)',u_dist(1:train));

tmp=find(lda_u==u_dist(train+1:end));
tmp2=find(lda_b==b_dist(train+1:end));
tmp3=find(lda_up==u_dist(train+1:end));
tmp4=find(lda_bp==b_dist(train+1:end));
test_rate_u=length(tmp)/length(u_dist(train+1:end));
test_rate_b=length(tmp2)/length(u_dist(train+1:end));
test_rate_up=length(tmp3)/length(u_dist(train+1:end));
test_rate_bp=length(tmp4)/length(u_dist(train+1:end));

figure; subplot(221)
supertitle('20°, 70n, 3k trials, poisson, R 70ms, ampl 25')
title('TC Model, all trials/all units (narrow bw, wide kern)')
hold on
plot(1-lda_err_u,test_rate_u,'k.','MarkerSize',10)
plot(1-lda_err_up,test_rate_up,'b.','MarkerSize',10)
plot(1-lda_err_b,test_rate_b,'r.','MarkerSize',10)
plot(1-lda_err_bp,test_rate_bp,'g.','MarkerSize',10)
ylim([0 1]); xlim([0 1])
xlabel('training: % correct')
ylabel('test: % correct')
legend({'UTC/Ustats','BTC/Ustats','BTC/Bstats','UTC/Bstats'},'Location','southeast')
axis square; box off
set(gca,'TickDir','out')
refline(1,0)
clear tmp* temp*

% [lda_u_w, lda_err_u_w]=classify(resp_u_ww(:,train+1:end)',resp_u_ww(:,1:train)',u_dist(1:train));
% [lda_b_w, lda_err_b_w]=classify(resp_ba_ww(:,train+1:end)',resp_ba_ww(:,1:train)',b_dist(1:train));
% [lda_up_w, lda_err_up_w]=classify(resp_ua_ww(:,train+1:end)',resp_ua_ww(:,1:train)',u_dist(1:train));
% [lda_bp_w, lda_err_bp_w]=classify(resp_b_ww(:,train+1:end)',resp_b_ww(:,1:train)',b_dist(1:train));
[lda_u_w, lda_err_u_w]=classify(resp_u_ww(:,train+1:end)',resp_u_ww(:,1:train)',u_dist(1:train));
[lda_b_w, lda_err_b_w]=classify(resp_ba_ww(:,train+1:end)',resp_ua_ww(:,1:train)',u_dist(1:train));
[lda_up_w, lda_err_up_w]=classify(resp_ua_ww(:,train+1:end)',resp_ua_ww(:,1:train)',u_dist(1:train));
[lda_bp_w, lda_err_bp_w]=classify(resp_b_ww(:,train+1:end)',resp_u_ww(:,1:train)',u_dist(1:train));

tmp=find(lda_u_w==u_dist(train+1:end));
tmp2=find(lda_b_w==b_dist(train+1:end));
tmp3=find(lda_up_w==u_dist(train+1:end));
tmp4=find(lda_bp_w==b_dist(train+1:end));
test_rate_u_w=length(tmp)/length(u_dist(train+1:end));
test_rate_b_w=length(tmp2)/length(u_dist(train+1:end));
test_rate_up_w=length(tmp3)/length(u_dist(train+1:end));
test_rate_bp_w=length(tmp4)/length(u_dist(train+1:end));

subplot(222)
title('TC Model, all trials/all units (wide bw, wide kern)')
hold on
plot(1-lda_err_u_w,test_rate_u_w,'k.','MarkerSize',10)
plot(1-lda_err_up_w,test_rate_up_w,'b.','MarkerSize',10)
plot(1-lda_err_b_w,test_rate_b_w,'r.','MarkerSize',10)
plot(1-lda_err_bp_w,test_rate_bp_w,'g.','MarkerSize',10)
ylim([0 1]); xlim([0 1])
xlabel('training: % correct')
ylabel('test: % correct')
axis square; box off
set(gca,'TickDir','out')
refline(1,0)

clear tmp* temp*

% [lda_u2, lda_err_u2]=classify(resp_u_nn(:,train+1:end)',resp_u_nn(:,1:train)',u_dist(1:train));
% [lda_b2, lda_err_b2]=classify(resp_ba_nn(:,train+1:end)',resp_ba_nn(:,1:train)',b_dist(1:train));
% [lda_up2, lda_err_up2]=classify(resp_ua_nn(:,train+1:end)',resp_ua_nn(:,1:train)',u_dist(1:train));
% [lda_bp2, lda_err_bp2]=classify(resp_b_nn(:,train+1:end)',resp_b_nn(:,1:train)',b_dist(1:train));
[lda_u2, lda_err_u2]=classify(resp_u_nn(:,train+1:end)',resp_u_nn(:,1:train)',u_dist(1:train));
[lda_b2, lda_err_b2]=classify(resp_ba_nn(:,train+1:end)',resp_ua_nn(:,1:train)',u_dist(1:train));
[lda_up2, lda_err_up2]=classify(resp_ua_nn(:,train+1:end)',resp_ua_nn(:,1:train)',u_dist(1:train));
[lda_bp2, lda_err_bp2]=classify(resp_b_nn(:,train+1:end)',resp_u_nn(:,1:train)',u_dist(1:train));

tmp=find(lda_u2==u_dist(train+1:end));
tmp2=find(lda_b2==b_dist(train+1:end));
tmp3=find(lda_up2==u_dist(train+1:end));
tmp4=find(lda_bp2==b_dist(train+1:end));
test_rate_u2=length(tmp)/length(u_dist(train+1:end));
test_rate_b2=length(tmp2)/length(u_dist(train+1:end));
test_rate_up2=length(tmp3)/length(u_dist(train+1:end));
test_rate_bp2=length(tmp4)/length(u_dist(train+1:end));

subplot(223)
title('TC Model, all trials/all units (narrow bw, narrow kern)')
hold on
plot(1-lda_err_u2,test_rate_u2,'k.','MarkerSize',10)
plot(1-lda_err_up2,test_rate_up2,'b.','MarkerSize',10)
plot(1-lda_err_b2,test_rate_b2,'r.','MarkerSize',10)
plot(1-lda_err_bp2,test_rate_bp2,'g.','MarkerSize',10)
ylim([0 1]); xlim([0 1])
xlabel('training: % correct')
ylabel('test: % correct')
axis square; box off
set(gca,'TickDir','out')
refline(1,0)
clear tmp* temp*

% [lda_u_w2, lda_err_u_w2]=classify(resp_u_wn(:,train+1:end)',resp_u_wn(:,1:train)',u_dist(1:train));
% [lda_b_w2, lda_err_b_w2]=classify(resp_ba_wn(:,train+1:end)',resp_ba_wn(:,1:train)',b_dist(1:train));
% [lda_up_w2, lda_err_up_w2]=classify(resp_ua_wn(:,train+1:end)',resp_ua_wn(:,1:train)',u_dist(1:train));
% [lda_bp_w2, lda_err_bp_w2]=classify(resp_b_wn(:,train+1:end)',resp_b_wn(:,1:train)',b_dist(1:train));
[lda_u_w2, lda_err_u_w2]=classify(resp_u_wn(:,train+1:end)',resp_u_wn(:,1:train)',u_dist(1:train));
[lda_b_w2, lda_err_b_w2]=classify(resp_ba_wn(:,train+1:end)',resp_ua_wn(:,1:train)',u_dist(1:train));
[lda_up_w2, lda_err_up_w2]=classify(resp_ua_wn(:,train+1:end)',resp_ua_wn(:,1:train)',u_dist(1:train));
[lda_bp_w2, lda_err_bp_w2]=classify(resp_b_wn(:,train+1:end)',resp_u_wn(:,1:train)',u_dist(1:train));

tmp=find(lda_u_w2==u_dist(train+1:end));
tmp2=find(lda_b_w2==b_dist(train+1:end));
tmp3=find(lda_up_w2==u_dist(train+1:end));
tmp4=find(lda_bp_w2==b_dist(train+1:end));
test_rate_u_w2=length(tmp)/length(u_dist(train+1:end));
test_rate_b_w2=length(tmp2)/length(u_dist(train+1:end));
test_rate_up_w2=length(tmp3)/length(u_dist(train+1:end));
test_rate_bp_w2=length(tmp4)/length(u_dist(train+1:end));

subplot(224)
title('TC Model, all trials/all units (wide bw, narrow kern)')
hold on
plot(1-lda_err_u_w2,test_rate_u_w2,'k.','MarkerSize',10)
plot(1-lda_err_up_w2,test_rate_up_w2,'b.','MarkerSize',10)
plot(1-lda_err_b_w2,test_rate_b_w2,'r.','MarkerSize',10)
plot(1-lda_err_bp_w2,test_rate_bp_w2,'g.','MarkerSize',10)
ylim([0 1]); xlim([0 1])
xlabel('training: % correct')
ylabel('test: % correct')
axis square; box off
set(gca,'TickDir','out')
refline(1,0)

clear tmp* temp*

% % sanity check: shuffle training set oris - result should be chance (1/9)
% tmp=randperm(train);
% [lda_us, lda_err_us]=classify(resp_u(:,train+1:end)',resp_u(:,tmp)',u_dist(1:train)');
% [lda_bs, lda_err_bs]=classify(resp_ba(:,train+1:end)',resp_ba(:,tmp)',b_dist(1:train)');
% [lda_ups, lda_err_ups]=classify(resp_ua(:,train+1:end)',resp_ua(:,tmp)',u_dist(1:train)');
% [lda_bps, lda_err_bps]=classify(resp_b(:,train+1:end)',resp_b(:,tmp)',b_dist(1:train)');
% 
% tmp=find(lda_us==u_dist(train+1:end));
% tmp2=find(lda_bs==b_dist(train+1:end));
% tmp3=find(lda_ups==u_dist(train+1:end));
% tmp4=find(lda_bps==b_dist(train+1:end));
% test_rate_us=length(tmp)/length(u_dist(train+1:end));
% test_rate_bs=length(tmp2)/length(u_dist(train+1:end));
% test_rate_ups=length(tmp3)/length(u_dist(train+1:end));
% test_rate_bps=length(tmp4)/length(u_dist(train+1:end));
% 
% figure
% title('TC Model, shuffled trials, classify all trials/all units (LDA)')
% hold on
% plot(1-lda_err_us,test_rate_us,'k.','MarkerSize',10)
% plot(1-lda_err_ups,test_rate_ups,'b.','MarkerSize',10)
% plot(1-lda_err_bs,test_rate_bs,'r.','MarkerSize',10)
% plot(1-lda_err_bps,test_rate_bps,'g.','MarkerSize',10)
% ylim([0 1]); xlim([0 1])
% xlabel('training: % correct')
% ylabel('test: % correct')
% legend({'Uniform','U pred','Bias','B pred'},'Location','southeast')
% axis square; box off
% set(gca,'TickDir','out')
% refline(1,0)
% disp('full done')

figure;
subplot(221); hold on
histogram(resp_u_nw)
histogram(resp_ba_nw)
axis square; box off
subplot(222); hold on
histogram(resp_u_ww)
histogram(resp_ba_ww)
axis square; box off
subplot(223); hold on
histogram(resp_u_nn)
histogram(resp_ba_nn)
axis square; box off
subplot(224); hold on
histogram(resp_u_wn)
histogram(resp_ba_wn)
axis square; box off
%% performance of each class vs its neighbors (3 classes)
clear kern* aktune* resp*
% wide kernel:  using strong (0.5) kernel
kernelw=0.1+exp(2*(cos(deg2rad(oris_fine-90))-1)); % kernel centered at 90
kernelw=1-0.5*kernelw;
% narrow kernel:
kerneln=0.1+exp(10*(cos(deg2rad(oris_fine-90))-1)); % kernel centered at 90
kerneln=1-0.5*kerneln;

for i=1:neurons
    % wide bw:
    aktune_ww(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(deg2rad(oris_fine-ori_prefs(i)))-1));
    aktune_a_ww(i,:)=aktune_ww(i,:).*kernelw;
    aktune_wn(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(deg2rad(oris_fine-ori_prefs(i)))-1));
    aktune_a_wn(i,:)=aktune_wn(i,:).*kerneln;
    % narrow bw:
    aktune_nw(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(deg2rad(oris_fine-ori_prefs(i)))-1));
    aktune_a_nw(i,:)=aktune_nw(i,:).*kernelw;
    aktune_nn(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(deg2rad(oris_fine-ori_prefs(i)))-1));
    aktune_a_nn(i,:)=aktune_nn(i,:).*kerneln;
end
% divide by 14 to approximate FR of 70ms epoch
% aktune_ww=aktune_ww./14;
% aktune_a_ww=aktune_a_ww./14;
% aktune_wn=aktune_wn./14;
% aktune_a_wn=aktune_a_wn./14;
% aktune_nw=aktune_nw./14;
% aktune_a_nw=aktune_a_nw./14;
% aktune_nn=aktune_nn./14;
% aktune_a_nn=aktune_a_nn./14;

lda_neighbors_u=nan*zeros(20,9,2); %30 repeats of sampled units x 9 oris/neighbors x [training test]
lda_neighbors_b=nan*zeros(20,9,2);
lda_neighbors_u2=nan*zeros(20,9,2); %30 repeats of sampled units x 9 oris/neighbors x [training test]
lda_neighbors_b2=nan*zeros(20,9,2);
lda_neighbors_u_w=nan*zeros(20,9,2); %30 repeats of sampled units x 9 oris/neighbors x [training test]
lda_neighbors_b_w=nan*zeros(20,9,2);
lda_neighbors_u_w2=nan*zeros(20,9,2); %30 repeats of sampled units x 9 oris/neighbors x [training test]
lda_neighbors_b_w2=nan*zeros(20,9,2);
% find and store each ori and its responses
for e = 1:length(oris_coarse)
    clear near_dist resp_*
    near_dist=[ones(1,trial_match*.9) 2*ones(1,trial_match*.9) 3*ones(1,trial_match*.9)]; % training trials
    near_dist=[near_dist ones(1,trial_match*.1) 2*ones(1,trial_match*.1) 3*ones(1,trial_match*.1)]; % training and test
    near_dist=near_dist+3+(e-1)*8;
%     near_dist=near_dist+1+(e-1)*4;
    
    for i=1:neurons
        resp_u_nw(i,:)=(poissrnd(aktune_nw(i,near_dist)));
        resp_ba_nw(i,:)=(poissrnd(aktune_a_nw(i,near_dist)));
        resp_u_ww(i,:)=(poissrnd(aktune_ww(i,near_dist)));
        resp_ba_ww(i,:)=(poissrnd(aktune_a_ww(i,near_dist)));
        resp_u_nn(i,:)=(poissrnd(aktune_nn(i,near_dist)));
        resp_ba_nn(i,:)=(poissrnd(aktune_a_nn(i,near_dist)));
        resp_u_wn(i,:)=(poissrnd(aktune_wn(i,near_dist)));
        resp_ba_wn(i,:)=(poissrnd(aktune_a_wn(i,near_dist)));
        
%         resp_u_nw(i,:)=((aktune_nw(i,near_dist)));
%         resp_ba_nw(i,:)=((aktune_a_nw(i,near_dist)));
%         resp_u_ww(i,:)=((aktune_ww(i,near_dist)));
%         resp_ba_ww(i,:)=((aktune_a_ww(i,near_dist)));
%         resp_u_nn(i,:)=((aktune_nn(i,near_dist)));
%         resp_ba_nn(i,:)=((aktune_a_nn(i,near_dist)));
%         resp_u_wn(i,:)=((aktune_wn(i,near_dist)));
%         resp_ba_wn(i,:)=((aktune_a_wn(i,near_dist)));
    end
    tmpsize=round(0.9*length(near_dist));
    
    for b=1:9
        n=randperm(neurons,36); % random sample from population of neurons
%         n=(3:7)+(8*(b-1));        % shifting window of population by ori preference
%         n(n>70)=1;
        clear tmp tmp1 tmp2 tmp3 tmp4
        % narrow bw, wide kernel:
        units_u=resp_u_nw(n,:);
        units_b=resp_ba_nw(n,:);
        [lda_near_u, lda_near_err_u]=classify(units_u(:,tmpsize+1:end)',units_u(:,1:tmpsize)',near_dist(1:tmpsize));
        [lda_near_b, lda_near_err_b]=classify(units_b(:,tmpsize+1:end)',units_b(:,1:tmpsize)',near_dist(1:tmpsize));
        tmp=find(lda_near_u'==near_dist(tmpsize+1:end));
        tmp2=find(lda_near_b'==near_dist(tmpsize+1:end));
        lda_neighbors_u(b,e,:)=[1-lda_near_err_u length(tmp)/length(lda_near_u)];
        lda_neighbors_b(b,e,:)=[1-lda_near_err_b length(tmp2)/length(lda_near_b)];
%         units_up=resp_ua(n,x);
%         units_bp=resp_b(n,y);
%         [lda_near_up, lda_near_err_up]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',u_dist(x(1:tmpsize)));
%         [lda_near_bp, lda_near_err_bp]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',b_dist(y(1:tmpsize2)));
%         tmp3=find(lda_near_up==u_dist(x(tmpsize+1:end)));
%         tmp4=find(lda_near_bp==b_dist(y(tmpsize2+1:end)));
%         lda_neighbors_up(b,e,:)=[1-lda_near_err_up length(tmp3)/length(lda_near_u)];
%         lda_neighbors_bp(b,e,:)=[1-lda_near_err_bp length(tmp4)/length(lda_near_b)];
        
        clear tmp tmp1 tmp2 tmp3 tmp4
        % wide bw, wide kernel:
        units_u=resp_u_ww(n,:);
        units_b=resp_ba_ww(n,:);
        [lda_near_u_w, lda_near_err_u_w]=classify(units_u(:,tmpsize+1:end)',units_u(:,1:tmpsize)',near_dist(1:tmpsize));
        [lda_near_b_w, lda_near_err_b_w]=classify(units_b(:,tmpsize+1:end)',units_b(:,1:tmpsize)',near_dist(1:tmpsize));
        tmp=find(lda_near_u_w'==near_dist(tmpsize+1:end));
        tmp2=find(lda_near_b_w'==near_dist(tmpsize+1:end));
        lda_neighbors_u_w(b,e,:)=[1-lda_near_err_u_w length(tmp)/length(lda_near_u_w)];
        lda_neighbors_b_w(b,e,:)=[1-lda_near_err_b_w length(tmp2)/length(lda_near_b_w)];
%         units_up=resp_ua_w(n,x);
%         units_bp=resp_b_w(n,y);
%         [lda_near_up_w, lda_near_err_up_w]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',u_dist(x(1:tmpsize)));
%         [lda_near_bp_w, lda_near_err_bp_w]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',b_dist(y(1:tmpsize2)));
%         tmp3=find(lda_near_up_w==u_dist(x(tmpsize+1:end)));
%         tmp4=find(lda_near_bp_w==b_dist(y(tmpsize2+1:end)));
%         lda_neighbors_up_w(b,e,:)=[1-lda_near_err_up_w length(tmp3)/length(lda_near_u_w)];
%         lda_neighbors_bp_w(b,e,:)=[1-lda_near_err_bp_w length(tmp4)/length(lda_near_b_w)];
        
        clear tmp tmp1 tmp2 tmp3 tmp4
        % narrow bw, narrow kernel:
        units_u=resp_u_nn(n,:);
        units_b=resp_ba_nn(n,:);
        [lda_near_u2, lda_near_err_u2]=classify(units_u(:,tmpsize+1:end)',units_u(:,1:tmpsize)',near_dist(1:tmpsize));
        [lda_near_b2, lda_near_err_b2]=classify(units_b(:,tmpsize+1:end)',units_b(:,1:tmpsize)',near_dist(1:tmpsize));
        tmp=find(lda_near_u2'==near_dist(tmpsize+1:end));
        tmp2=find(lda_near_b2'==near_dist(tmpsize+1:end));
        lda_neighbors_u2(b,e,:)=[1-lda_near_err_u2 length(tmp)/length(lda_near_u2)];
        lda_neighbors_b2(b,e,:)=[1-lda_near_err_b2 length(tmp2)/length(lda_near_b2)];
%         units_up=resp_ua2(n,x);
%         units_bp=resp_b2(n,y);
%         [lda_near_up2, lda_near_err_up2]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',u_dist(x(1:tmpsize)));
%         [lda_near_bp2, lda_near_err_bp2]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',b_dist(y(1:tmpsize2)));
%         tmp3=find(lda_near_up2==u_dist(x(tmpsize+1:end)));
%         tmp4=find(lda_near_bp2==b_dist(y(tmpsize2+1:end)));
%         lda_neighbors_up2(b,e,:)=[1-lda_near_err_up2 length(tmp3)/length(lda_near_u2)];
%         lda_neighbors_bp2(b,e,:)=[1-lda_near_err_bp2 length(tmp4)/length(lda_near_b2)];
        
        clear tmp tmp1 tmp2 tmp3 tmp4
        % wide bw, narrow kernel:
        units_u=resp_u_wn(n,:);
        units_b=resp_ba_wn(n,:);
        [lda_near_u_w2, lda_near_err_u_w2]=classify(units_u(:,tmpsize+1:end)',units_u(:,1:tmpsize)',near_dist(1:tmpsize));
        [lda_near_b_w2, lda_near_err_b_w2]=classify(units_b(:,tmpsize+1:end)',units_b(:,1:tmpsize)',near_dist(1:tmpsize));
        tmp=find(lda_near_u_w2'==near_dist(tmpsize+1:end));
        tmp2=find(lda_near_b_w2'==near_dist(tmpsize+1:end));
        lda_neighbors_u_w2(b,e,:)=[1-lda_near_err_u_w2 length(tmp)/length(lda_near_u_w2)];
        lda_neighbors_b_w2(b,e,:)=[1-lda_near_err_b_w2 length(tmp2)/length(lda_near_b_w2)];
%         units_up=resp_ua_w2(n,x);
%         units_bp=resp_b_w2(n,y);
%         [lda_near_up_w2, lda_near_err_up_w2]=classify(units_up(:,tmpsize+1:end)',units_up(:,1:tmpsize)',u_dist(x(1:tmpsize)));
%         [lda_near_bp_w2, lda_near_err_bp_w2]=classify(units_bp(:,tmpsize2+1:end)',units_bp(:,1:tmpsize2)',b_dist(y(1:tmpsize2)));
%         tmp3=find(lda_near_up_w2==u_dist(x(tmpsize+1:end)));
%         tmp4=find(lda_near_bp_w2==b_dist(y(tmpsize2+1:end)));
%         lda_neighbors_up_w2(b,e,:)=[1-lda_near_err_up_w2 length(tmp3)/length(lda_near_u_w2)];
%         lda_neighbors_bp_w2(b,e,:)=[1-lda_near_err_bp_w2 length(tmp4)/length(lda_near_b_w2)];
    end
end

figure
subplot(221); hold on
supertitle('Training performance - neighbors - 30 neurons, 2.5°')
title('narrow bw, wide kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_u(:,:,1),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_b(:,:,1),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
legend({'unif','bias'},'Location','northeast')
xlabel('center ori +/-20')
ylabel('% correct')
subplot(222); hold on
title('wide bw, wide kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_w(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_u_w(:,:,1),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_w(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_b_w(:,:,1),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')
subplot(223); hold on
title('narrow bw, narrow kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u2(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_u2(:,:,1),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b2(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_b2(:,:,1),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')
subplot(224); hold on
title('wide bw, narrow kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_w2(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_u_w2(:,:,1),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_w2(:,:,1),1)),0),circshift(squeeze(nanstd(lda_neighbors_b_w2(:,:,1),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')

figure
subplot(231); hold on
supertitle('Test performance - neighbors - 30 neurons, 2.5°')
title('narrow bw, wide kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_u(:,:,2),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_b(:,:,2),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
legend({'unif','bias'},'Location','northeast')
xlabel('center ori +/-20')
ylabel('% correct')
subplot(232); hold on
title('wide bw, wide kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_w(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_u_w(:,:,2),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_w(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_b_w(:,:,2),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')
subplot(234); hold on
title('narrow bw, narrow kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u2(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_u2(:,:,2),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b2(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_b2(:,:,2),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')
subplot(235); hold on
title('wide bw, narrow kern')
errorline(1:9,circshift(squeeze(nanmean(lda_neighbors_u_w2(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_u_w2(:,:,2),1)),0),'k')
errorline(1.1:1:9.1,circshift(squeeze(nanmean(lda_neighbors_b_w2(:,:,2),1)),0),circshift(squeeze(nanstd(lda_neighbors_b_w2(:,:,2),1)),0),'r')
ylim([0 1])
xlim([0 10])
set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('center ori +/-20')
ylabel('% correct')
subplot(233)
plot(kernelw)
axis square; box off
subplot(236)
plot(kerneln)
axis square; box off
stop
% % % % % % all populations on specific orientations
figure
subplot(221); hold on
supertitle('Test performance - neighbors - 30 neurons, 2.5°')
title('narrow bw, wide kern')
plot(lda_neighbors_u(:,1,1),lda_neighbors_u(:,1,2),'k.')
plot(lda_neighbors_u(:,3,1),lda_neighbors_u(:,3,2),'b.')
plot(lda_neighbors_u(:,5,1),lda_neighbors_u(:,5,2),'r.')
ylim([0 1]);xlim([0 1])
refline(1,0)
axis square; box off
legend({'80','40','0'},'Location','northeast')
xlabel('% correct, train')
ylabel('% correct, test')
subplot(222); hold on
title('wide bw, wide kern')
plot(lda_neighbors_u_w(:,1,1),lda_neighbors_u_w(:,1,2),'k.')
plot(lda_neighbors_u_w(:,3,1),lda_neighbors_u_w(:,3,2),'b.')
plot(lda_neighbors_u_w(:,5,1),lda_neighbors_u_w(:,5,2),'r.')
ylim([0 1]);xlim([0 1])
refline(1,0)
axis square; box off
subplot(223); hold on
title('narrow bw, narrow kern')
plot(lda_neighbors_u2(:,1,1),lda_neighbors_u2(:,1,2),'k.')
plot(lda_neighbors_u2(:,3,1),lda_neighbors_u2(:,3,2),'b.')
plot(lda_neighbors_u2(:,5,1),lda_neighbors_u2(:,5,2),'r.')
ylim([0 1]);xlim([0 1])
axis square; box off
refline(1,0)
subplot(224); hold on
title('wide bw, narrow kern')
plot(lda_neighbors_u_w2(:,1,1),lda_neighbors_u_w2(:,1,2),'k.')
plot(lda_neighbors_u_w2(:,3,1),lda_neighbors_u_w2(:,3,2),'b.')
plot(lda_neighbors_u_w2(:,5,1),lda_neighbors_u_w2(:,5,2),'r.')
ylim([0 1]);xlim([0 1])
axis square; box off
refline(1,0)
% Performance does not seem to be stimulus specific

% % % % % % specific populations on all orientations
figure
subplot(221); hold on
% supertitle('Test performance - neighbors - 30 neurons, 2.5°')
title('narrow bw, wide kern')
plot(lda_neighbors_u(1,:,1),lda_neighbors_u(1,:,2),'k.')
plot(lda_neighbors_u(6,:,1),lda_neighbors_u(6,:,2),'g.')
plot(lda_neighbors_u(11,:,1),lda_neighbors_u(11,:,2),'r.')
ylim([0 1])
refline(1,0)
% xlim([0 10])
% set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
legend({'bias','40','80'},'Location','northeast')
xlabel('% correct train')
ylabel('% correct test')
subplot(222); hold on
title('wide bw, wide kern')
plot(lda_neighbors_u_w(1,:,1),lda_neighbors_u_w(1,:,2),'k.')
plot(lda_neighbors_u_w(6,:,1),lda_neighbors_u_w(6,:,2),'g.')
plot(lda_neighbors_u_w(11,:,1),lda_neighbors_u_w(11,:,2),'r.')
ylim([0 1])
refline(1,0)
% xlim([0 10])
% set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('% correct train')
ylabel('% correct test')
subplot(223); hold on
title('narrow bw, narrow kern')
plot(lda_neighbors_u2(1,:,1),lda_neighbors_u2(1,:,2),'k.')
plot(lda_neighbors_u2(6,:,1),lda_neighbors_u2(6,:,2),'g.')
plot(lda_neighbors_u2(11,:,1),lda_neighbors_u2(11,:,2),'r.')
ylim([0 1])
refline(1,0)
% xlim([0 10])
% set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('% correct train')
ylabel('% correct test')
subplot(224); hold on
title('wide bw, narrow kern')
plot(lda_neighbors_u_w2(1,:,1),lda_neighbors_u_w2(1,:,2),'k.')
plot(lda_neighbors_u_w2(6,:,1),lda_neighbors_u_w2(6,:,2),'g.')
plot(lda_neighbors_u_w2(11,:,1),lda_neighbors_u_w2(11,:,2),'r.')
ylim([0 1])
refline(1,0)
% xlim([0 10])
% set(gca,'XTick',1:9,'XTickLabel',{'-80','-60','-40','-20','0','20','40','60','80'},'TickDir','out')
axis square; box off
xlabel('% correct train')
ylabel('% correct test')
% performance is neuron-specific. Each does best near it's orientation preference
stop

% only run this for shifted window of units selected
figure
subplot(4,4,1)
imagesc(lda_neighbors_u(1:9,:,1),[0.1 0.9])
ylabel('Test Oris')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'})
title('u train')
subplot(4,4,2)
imagesc(lda_neighbors_u(1:9,:,2),[0.1 0.9])
title('u test')
subplot(4,4,5)
imagesc(lda_neighbors_b(1:9,:,1),[0.1 0.9])
ylabel('Test Oris')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'})
title('b train')
subplot(4,4,6)
imagesc(lda_neighbors_b(1:9,:,2),[0.1 0.9])
title('b test')

subplot(4,4,3)
imagesc(lda_neighbors_u2(1:9,:,1),[0.1 0.9])
title('u2 train')
subplot(4,4,4)
imagesc(lda_neighbors_u2(1:9,:,2),[0.1 0.9])
title('u2 test')
subplot(4,4,7)
imagesc(lda_neighbors_b2(1:9,:,1),[0.1 0.9])
title('b2 train')
subplot(4,4,8)
imagesc(lda_neighbors_b2(1:9,:,2),[0.1 0.9])
title('b2 test')

subplot(4,4,9)
imagesc(lda_neighbors_u_w(1:9,:,1),[0.1 0.9])
ylabel('Test Oris')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'})
title('uw train')
subplot(4,4,10)
imagesc(lda_neighbors_u_w(1:9,:,2),[0.1 0.9])
title('uw test')
subplot(4,4,13)
imagesc(lda_neighbors_b_w(1:9,:,1),[0.1 0.9])
ylabel('Test Oris')
xlabel('Ori Pref Bin')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'},...
    'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
title('bw train')
subplot(4,4,14)
imagesc(lda_neighbors_b_w(1:9,:,2),[0.1 0.9])
xlabel('Ori Pref Bin')
set(gca,'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
title('bw test')

subplot(4,4,11)
imagesc(lda_neighbors_u_w2(1:9,:,1),[0.1 0.9])
title('uw2 train')
subplot(4,4,12)
imagesc(lda_neighbors_u_w2(1:9,:,2),[0.1 0.9])
title('uw2 test')
subplot(4,4,15)
imagesc(lda_neighbors_b_w2(1:9,:,1),[0.1 0.9])
xlabel('Ori Pref Bin')
set(gca,'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
title('bw2 train')
subplot(4,4,16)
imagesc(lda_neighbors_b_w2(1:9,:,2),[0.1 0.9])
xlabel('Ori Pref Bin')
set(gca,'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
title('bw2 test')
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
figure
supertitle('Test performance diff (bias-uniform)')
subplot(2,2,1)
imagesc(lda_neighbors_b(1:9,:,2)-lda_neighbors_u(1:9,:,2),[-.25 .25])
ylabel('Test Oris')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'})
title('narrow bw, narrow kern')
subplot(2,2,2)
imagesc(lda_neighbors_b2(1:9,:,2)-lda_neighbors_u2(1:9,:,2),[-.25 .25])
title('wide bw, narrow kern')
subplot(2,2,3)
imagesc(lda_neighbors_b_w(1:9,:,2)-lda_neighbors_u_w(1:9,:,2),[-.25 .25])
ylabel('Test Oris')
xlabel('Ori Pref Bin')
set(gca,'YTick',[1 5 9],'YTickLabel',{'10+/-2.5','90+/-2.5','170+/-2.5'},...
    'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
title('wide bw, wide kern')
subplot(2,2,4)
imagesc(lda_neighbors_b_w2(1:9,:,2)-lda_neighbors_u_w2(1:9,:,2),[-.25 .25])
title('narrow bw, wide kern')
xlabel('Ori Pref Bin')
set(gca,'XTick',1:9,'XTickLabel',{'10','30','50','70','90','110','130','150','170'})
stop
%% performance w/ different # of neurons
pop_err_u=nan*zeros(20,neurons-1);
pop_err_b=nan*zeros(20,neurons-1);
pop_err_up=nan*zeros(20,neurons-1);
pop_err_bp=nan*zeros(20,neurons-1);
pop_class_u=nan*zeros(20,neurons-1,0.1*length(b_dist));
pop_class_b=nan*zeros(20,neurons-1,0.1*length(b_dist));
pop_class_up=nan*zeros(20,neurons-1,0.1*length(b_dist));
pop_class_bp=nan*zeros(20,neurons-1,0.1*length(b_dist));
for a=1:20
    for e=2:neurons
        j=randperm(neurons,e);
        units_u=resp_u(j,:);
        units_b=resp_ba(j,:);
        units_up=resp_ua(j,:);
        units_bp=resp_b(j,:);
        
        [tmp_lda_u,tmp_err_u]=classify(units_u(:,train+1:end)',units_u(:,1:train)',u_dist(1:train));
        [tmp_lda_b,tmp_err_b]=classify(units_b(:,train+1:end)',units_b(:,1:train)',b_dist(1:train));
        [tmp_lda_up,tmp_err_up]=classify(units_up(:,train+1:end)',units_up(:,1:train)',u_dist(1:train));
        [tmp_lda_bp,tmp_err_bp]=classify(units_bp(:,train+1:end)',units_bp(:,1:train)',b_dist(1:train));
        pop_err_u(a,e-1)=tmp_err_u;
        pop_err_b(a,e-1)=tmp_err_b;
        pop_err_up(a,e-1)=tmp_err_up;
        pop_err_bp(a,e-1)=tmp_err_bp;
        pop_class_u(a,e-1,:)=tmp_lda_u;
        pop_class_b(a,e-1,:)=tmp_lda_b;
        pop_class_up(a,e-1,:)=tmp_lda_up;
        pop_class_bp(a,e-1,:)=tmp_lda_bp;
    end
end
figure
supertitle('TC model, LDA performance vs population size (all trials)')
subplot(221); hold on
title('Training')
errorline(2:neurons,1-mean(pop_err_u,1),std(pop_err_u,1),'k')
errorline(2:neurons,1-mean(pop_err_b,1),std(pop_err_b,1),'r')
ylim([0 1])
axis square; box off
legend({'unif','bias'})
xlabel('# units')
ylabel('% correct')
subplot(223); hold on
errorline(2:neurons,1-mean(pop_err_up,1),std(pop_err_up,1),'k--')
errorline(2:neurons,1-mean(pop_err_bp,1),std(pop_err_bp,1),'r--')
ylim([0 1])
title('Training predicted')
axis square; box off
xlabel('# units')
ylabel('% correct')
subplot(222); hold on
test_rate_u_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
test_rate_b_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
test_rate_bp_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
test_rate_up_pop=nan*zeros(size(pop_err_u,1),size(pop_err_u,2));
title('Test')
for a = 1:size(pop_err_u,1)
    for e = 1:size(pop_err_u,2)
        tmp=find(squeeze(pop_class_u(a,e,:))==u_dist(train+1:end));
        tmp2=find(squeeze(pop_class_b(a,e,:))==b_dist(train+1:end));
        tmp3=find(squeeze(pop_class_up(a,e,:))==u_dist(train+1:end));
        tmp4=find(squeeze(pop_class_bp(a,e,:))==b_dist(train+1:end));
        
        test_rate_u_pop(a,e)=length(tmp)/length(u_dist(train+1:end));
        test_rate_b_pop(a,e)=length(tmp2)/length(u_dist(train+1:end));
        test_rate_up_pop(a,e)=length(tmp3)/length(u_dist(train+1:end));
        test_rate_bp_pop(a,e)=length(tmp4)/length(u_dist(train+1:end));
    end
end
errorline(2:neurons,mean(test_rate_u_pop,1),std(test_rate_u_pop,1),'k')
errorline(2:neurons,mean(test_rate_b_pop,1),std(test_rate_b_pop,1),'r')
ylim([0 1])
axis square; box off
xlabel('# of units')
ylabel('% correct')
subplot(224); hold on
title('Test predicted')
errorline(2:neurons,mean(test_rate_up_pop,1),std(test_rate_up_pop,1),'k--')
errorline(2:neurons,mean(test_rate_bp_pop,1),std(test_rate_bp_pop,1),'r--')
ylim([0 1])
axis square; box off
disp('pop size done')
clear tmp* units*


%% performance w/ different # of trials
trial_err_u=nan*zeros(20,5);
trial_err_b=nan*zeros(20,5);
trial_class_u=cell(20,5);
trial_class_b=cell(20,5);
for a=1:20
    id=1;
    for e=[125 250 500 1000 2000]
        j=randperm(neurons,5);
        k=randperm(trials,e);
        units_u=resp_u(j,k);
        units_b=resp_ba(j,k);
        units_up=resp_ua(j,k);
        units_bp=resp_b(j,k);
        
        tenpercent=0.1*e;
        tmpsize=round(0.9*length(k));
        uo=u_dist(k);
        bo=b_dist(k);
        
        [tmp_lda_u,tmp_err_u]=classify(units_u(:,end-tenpercent+1:end)',units_u(:,1:end-tenpercent)',uo(1:end-tenpercent));
        [tmp_lda_b,tmp_err_b]=classify(units_b(:,end-tenpercent+1:end)',units_b(:,1:end-tenpercent)',bo(1:end-tenpercent));
        [tmp_lda_up,tmp_err_up]=classify(units_up(:,end-tenpercent+1:end)',units_up(:,1:end-tenpercent)',uo(1:end-tenpercent));
        [tmp_lda_bp,tmp_err_bp]=classify(units_bp(:,end-tenpercent+1:end)',units_bp(:,1:end-tenpercent)',bo(1:end-tenpercent));
        trial_err_u(a,id)=tmp_err_u;
        trial_err_b(a,id)=tmp_err_b;
        trial_err_up(a,id)=tmp_err_up;
        trial_err_bp(a,id)=tmp_err_bp;
        trial_class_u{a,id}=[tmp_lda_u uo(end-tenpercent+1:end)];
        trial_class_b{a,id}=[tmp_lda_b bo(end-tenpercent+1:end)];
        trial_class_up{a,id}=[tmp_lda_up uo(end-tenpercent+1:end)];
        trial_class_bp{a,id}=[tmp_lda_bp bo(end-tenpercent+1:end)];
        id=id+1;
    end
end
figure
supertitle('TC model, LDA performance vs trial size (10 neurons)')
subplot(121); hold on
title('Training')
errorline(1-nanmean(trial_err_u,1),nanstd(trial_err_u,1),'k')
errorline(1-nanmean(trial_err_b,1),nanstd(trial_err_b,1),'r')
% errorline(1-mean(trial_err_up,1),std(trial_err_up,1),'k--')
% errorline(1-mean(trial_err_bp,1),std(trial_err_bp,1),'r--')
ylim([0 1])
set(gca,'XTick',1:5,'XTickLabel',{'125','250','500','1000','2000'})
axis square; box off
legend({'unif','bias','u pred','b pred'})
xlabel('# trials included')
ylabel('% correct')

subplot(122); hold on
title('Test')
for a = 1:size(trial_err_u,1)
    for e = 1:size(trial_err_u,2)
        temp=squeeze(trial_class_u{a,e});
        tmp=find(temp(:,1)==temp(:,2));
        temp=squeeze(trial_class_b{a,e});
        tmp2=find(temp(:,1)==temp(:,2));
        temp=squeeze(trial_class_up{a,e});
        tmp3=find(temp(:,1)==temp(:,2));
        temp=squeeze(trial_class_bp{a,e});
        tmp4=find(temp(:,1)==temp(:,2));
        
        test_rate_u_trl(a,e)=length(tmp)/length(temp);
        test_rate_b_trl(a,e)=length(tmp2)/length(temp);
        test_rate_up_trl(a,e)=length(tmp3)/length(temp);
        test_rate_bp_trl(a,e)=length(tmp4)/length(temp);
    end
end
errorline(mean(test_rate_u_trl,1),std(test_rate_u_trl,1),'k')
errorline(mean(test_rate_b_trl,1),std(test_rate_b_trl,1),'r')
errorline(mean(test_rate_up_trl,1),std(test_rate_up_trl,1),'k--')
errorline(mean(test_rate_bp_trl,1),std(test_rate_bp_trl,1),'r--')
ylim([0 1])
set(gca,'XTick',1:5,'XTickLabel',{'125','250','500','1000','2000'})
axis square; box off
legend({'unif','bias','u pred','b pred'})
xlabel('# trials included')
ylabel('% correct')




clear e tmp* selected* j k id a units* temp* n1 n2 i ans bo uo dmi* ten* x y
d=date;
stop

%% single unit mutual information
clear

% set trials, pop size, and ori prefs/oris used
trials=6000;        % approximate avg of trials in awake data
neurons=70;         % approximate avg of # of units in awake data
% trial_match=300;    % approximate avg of trials used in neighbors classification in awake data
oris_coarse=10:20:170;   % oris used in experiment
% oris_fine=0:2.5:177.5;  % oris for neighbors classification
ori_prefs=0:180/neurons:180;

% create tuning curves: adam's way
ampl=11*rand(neurons,1);
base=1*rand(neurons,1);
% wide bandwidth:
tmp=rand(neurons,1)-0.5;
bww=2+5*tmp; % originally 3.5 + 8
bww(bww<0)=0.01;
% add a second, narrower bw
bwn=8+10*tmp;
bwn(bwn<0)=0.01;

% wide kernel:  using strong (0.5) kernel
kernelw=exp(2*(cos(2*deg2rad(oris_coarse-90))-1)); % kernel centered at 90
kernelw=1-0.5*kernelw;
% narrow kernel:
kerneln=exp(10*(cos(2*deg2rad(oris_coarse-90))-1)); % kernel centered at 90
kerneln=1-0.5*kerneln;

for i=1:neurons
    % wide bw:
    aktune_ww(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(2*deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_ww(i,:)=aktune_ww(i,:).*kernelw;
    aktune_wn(i,:)=base(i)+ampl(i)*exp(bww(i)*(cos(2*deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_wn(i,:)=aktune_wn(i,:).*kerneln;
    % narrow bw:
    aktune_nw(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(2*deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_nw(i,:)=aktune_nw(i,:).*kernelw;
    aktune_nn(i,:)=base(i)+ampl(i)*exp(bwn(i)*(cos(2*deg2rad(oris_coarse-ori_prefs(i)))-1));
    aktune_a_nn(i,:)=aktune_nn(i,:).*kerneln;
end
% divide by 14 to approximate FR of 70ms epoch
% aktune_ww=aktune_ww./14;
% aktune_a_ww=aktune_a_ww./14;
% aktune_wn=aktune_wn./14;
% aktune_a_wn=aktune_a_wn./14;
% aktune_nw=aktune_nw./14;
% aktune_a_nw=aktune_a_nw./14;
% aktune_nn=aktune_nn./14;
% aktune_a_nn=aktune_a_nn./14;

u_dist=randi(9,trials,1);
b_dist=randi(14,trials,1); % 14
b_dist(b_dist>9)=5;   % bias == 90

for i=1:neurons            
    resp_u_nw(i,:)=poissrnd(aktune_nw(i,u_dist));
    resp_b_nw(i,:)=poissrnd(aktune_nw(i,b_dist));
    resp_ua_nw(i,:)=poissrnd(aktune_a_nw(i,u_dist));
    resp_ba_nw(i,:)=poissrnd(aktune_a_nw(i,b_dist));
    resp_u_ww(i,:)=poissrnd(aktune_ww(i,u_dist));
    resp_b_ww(i,:)=poissrnd(aktune_ww(i,b_dist));
    resp_ua_ww(i,:)=poissrnd(aktune_a_ww(i,u_dist));
    resp_ba_ww(i,:)=poissrnd(aktune_a_ww(i,b_dist));
    
    resp_u_nn(i,:)=poissrnd(aktune_nn(i,u_dist));
    resp_b_nn(i,:)=poissrnd(aktune_nn(i,b_dist));
    resp_ua_nn(i,:)=poissrnd(aktune_a_nn(i,u_dist));
    resp_ba_nn(i,:)=poissrnd(aktune_a_nn(i,b_dist));
    resp_u_wn(i,:)=poissrnd(aktune_wn(i,u_dist));
    resp_b_wn(i,:)=poissrnd(aktune_wn(i,b_dist));
    resp_ua_wn(i,:)=poissrnd(aktune_a_wn(i,u_dist));
    resp_ba_wn(i,:)=poissrnd(aktune_a_wn(i,b_dist));
end

for i = 1:neurons
    binnw=(min([resp_u_nw(i,:) resp_b_nw(i,:) resp_ua_nw(i,:) resp_ba_nw(i,:)]):1:max([resp_u_nw(i,:) resp_b_nw(i,:) resp_ua_nw(i,:) resp_ba_nw(i,:)]));
    binnn=(min([resp_u_nn(i,:) resp_b_nn(i,:) resp_ua_nn(i,:) resp_ba_nn(i,:)]):1:max([resp_u_nn(i,:) resp_b_nn(i,:) resp_ua_nn(i,:) resp_ba_nn(i,:)]));
    binwn=(min([resp_u_wn(i,:) resp_b_wn(i,:) resp_ua_wn(i,:) resp_ba_wn(i,:)]):1:max([resp_u_wn(i,:) resp_b_wn(i,:) resp_ua_wn(i,:) resp_ba_wn(i,:)]));
    binww=(min([resp_u_ww(i,:) resp_b_ww(i,:) resp_ua_ww(i,:) resp_ba_ww(i,:)]):1:max([resp_u_ww(i,:) resp_b_ww(i,:) resp_ua_ww(i,:) resp_ba_ww(i,:)]));
    
    % 1d probability matrices:
    a1nw=histc(resp_u_nw(i,:),binnw)/trials;      %P(r) No A in uniform
    a2nw=histc(resp_ba_nw(i,:),binnw)/trials;     %P(r) Adapt in bias
    a3nw=histc(resp_ua_nw(i,:),binnw)/trials;     %P(r) Adapt in uniform
    a4nw=histc(resp_b_nw(i,:),binnw)/trials;      %P(r) no A in bias
    a1nn=histc(resp_u_nn(i,:),binnn)/trials; 
    a2nn=histc(resp_ba_nn(i,:),binnn)/trials;
    a3nn=histc(resp_ua_nn(i,:),binnn)/trials;
    a4nn=histc(resp_b_nn(i,:),binnn)/trials; 
    a1wn=histc(resp_u_wn(i,:),binwn)/trials; 
    a2wn=histc(resp_ba_wn(i,:),binwn)/trials;
    a3wn=histc(resp_ua_wn(i,:),binwn)/trials;
    a4wn=histc(resp_b_wn(i,:),binwn)/trials; 
    a1ww=histc(resp_u_ww(i,:),binww)/trials; 
    a2ww=histc(resp_ba_ww(i,:),binww)/trials;
    a3ww=histc(resp_ua_ww(i,:),binww)/trials;
    a4ww=histc(resp_b_ww(i,:),binww)/trials; 
    
    % calculate entropy of responses:
    for j=1:length(binnw)
        entr_u_nw(j)=a1nw(j)*log2(a1nw(j));
        entr_b_nw(j)=a2nw(j)*log2(a2nw(j));
        entr_u_pred_nw(j)=a3nw(j)*log2(a3nw(j));
        entr_b_pred_nw(j)=a4nw(j)*log2(a4nw(j));
    end
    for j=1:length(binnn)
        entr_u_nn(j)=a1nn(j)*log2(a1nn(j));
        entr_b_nn(j)=a2nn(j)*log2(a2nn(j));
        entr_u_pred_nn(j)=a3nn(j)*log2(a3nn(j));
        entr_b_pred_nn(j)=a4nn(j)*log2(a4nn(j));
    end
    for j=1:length(binwn)
        entr_u_wn(j)=a1wn(j)*log2(a1wn(j));
        entr_b_wn(j)=a2wn(j)*log2(a2wn(j));
        entr_u_pred_wn(j)=a3wn(j)*log2(a3wn(j));
        entr_b_pred_wn(j)=a4wn(j)*log2(a4wn(j));
    end
    for j=1:length(binww)
        entr_u_ww(j)=a1ww(j)*log2(a1ww(j));
        entr_b_ww(j)=a2ww(j)*log2(a2ww(j));
        entr_u_pred_ww(j)=a3ww(j)*log2(a3ww(j));
        entr_b_pred_ww(j)=a4ww(j)*log2(a4ww(j));
    end
    
    % calculate conditional entropy
    for j=1:length(oris_coarse)
        xxu=find(u_dist==j);     % cases of each ori
        u_xx_nw=histc(resp_u_nw(i,xxu),binnw)/length(xxu);    % P(r) in each bin for given ori
        u_xx_nn=histc(resp_u_nn(i,xxu),binnn)/length(xxu);
        u_xx_wn=histc(resp_u_wn(i,xxu),binwn)/length(xxu);
        u_xx_ww=histc(resp_u_ww(i,xxu),binww)/length(xxu);
        
        bp_xx_nw=histc(resp_b_nw(i,xxu),binnw)/length(xxu);
        bp_xx_nn=histc(resp_b_nn(i,xxu),binnn)/length(xxu);
        bp_xx_wn=histc(resp_b_wn(i,xxu),binwn)/length(xxu);
        bp_xx_ww=histc(resp_b_ww(i,xxu),binww)/length(xxu);
        
        p_s_base(j)=length(xxu)/length(u_dist);  % probability of each stimulus in uniform dist
        for k=1:length(binnw)
            cond_ent_base_nw(j,k)=u_xx_nw(k)*log2(u_xx_nw(k));   % conditional entropy for uniform
            cond_ent_bias_pred_nw(j,k)=bp_xx_nw(k)*log2(bp_xx_nw(k)); %cond. entropy for bias distribution
        end
        for k=1:length(binnn)
            cond_ent_base_nn(j,k)=u_xx_nn(k)*log2(u_xx_nn(k));
            cond_ent_bias_pred_nn(j,k)=bp_xx_nn(k)*log2(bp_xx_nn(k));
        end
        for k=1:length(binwn)
            cond_ent_base_wn(j,k)=u_xx_wn(k)*log2(u_xx_wn(k));
            cond_ent_bias_pred_wn(j,k)=bp_xx_wn(k)*log2(bp_xx_wn(k));
        end
        for k=1:length(binww)
            cond_ent_base_ww(j,k)=u_xx_ww(k)*log2(u_xx_ww(k));
            cond_ent_bias_pred_ww(j,k)=bp_xx_ww(k)*log2(bp_xx_ww(k));
        end

        xx=find(b_dist==j);
        b_xx_nw=histc(resp_ba_nw(i,xx),binnw)/length(xx);    % P(r) in each bin for given ori
        b_xx_nn=histc(resp_ba_nn(i,xx),binnn)/length(xx);
        b_xx_wn=histc(resp_ba_wn(i,xx),binwn)/length(xx);
        b_xx_ww=histc(resp_ba_ww(i,xx),binww)/length(xx);
        
        up_xx_nw=histc(resp_ua_nw(i,xx),binnw)/length(xx);
        up_xx_nn=histc(resp_ua_nn(i,xx),binnn)/length(xx);
        up_xx_wn=histc(resp_ua_wn(i,xx),binwn)/length(xx);
        up_xx_ww=histc(resp_ua_ww(i,xx),binww)/length(xx);
        
        p_s_bias(j)=length(xx)/length(b_dist);  % probability of each stimulus in bias dist
        for k=1:length(binnw)
            cond_ent_bias_nw(j,k)=b_xx_nw(k)*log2(b_xx_nw(k));   % conditional entropy for uniform
            cond_ent_base_pred_nw(j,k)=up_xx_nw(k)*log2(up_xx_nw(k)); %cond. entropy for bias pred
        end
        for k=1:length(binnn)
            cond_ent_bias_nn(j,k)=b_xx_nn(k)*log2(b_xx_nn(k));
            cond_ent_base_pred_nn(j,k)=up_xx_nn(k)*log2(up_xx_nn(k));
        end
        for k=1:length(binwn)
            cond_ent_bias_wn(j,k)=b_xx_wn(k)*log2(b_xx_wn(k));
            cond_ent_base_pred_wn(j,k)=up_xx_wn(k)*log2(up_xx_wn(k));
        end
        for k=1:length(binww)
            cond_ent_bias_ww(j,k)=b_xx_ww(k)*log2(b_xx_ww(k));
            cond_ent_base_pred_ww(j,k)=up_xx_ww(k)*log2(up_xx_ww(k));
        end        
    end
    H_base_nw(i)=-1*nansum(entr_u_nw);              % unadapted TC, uniform dist
    CondH_base_nw(i)=sum(p_s_base.*nansum(cond_ent_base_nw,2)'); 
    H_bias_nw(i)=-1*nansum(entr_b_nw);              % adapted TC, bias dist
    CondH_bias_nw(i)=sum(p_s_bias.*nansum(cond_ent_bias_nw,2)');  
    H_base_pred_nw(i)=-1*nansum(entr_u_pred_nw);    % Adapted TC, uniform dist
    CondH_base_pred_nw(i)=sum(p_s_base.*nansum(cond_ent_base_pred_nw,2)');
    H_bias_pred_nw(i)=-1*nansum(entr_b_pred_nw);    % unadapted TC, bias dist
    CondH_bias_pred_nw(i)=sum(p_s_bias.*nansum(cond_ent_bias_pred_nw,2)');
    
    H_base_nn(i)=-1*nansum(entr_u_nn);                          
    CondH_base_nn(i)=sum(p_s_base.*nansum(cond_ent_base_nn,2)');
    H_bias_nn(i)=-1*nansum(entr_b_nn);
    CondH_bias_nn(i)=sum(p_s_bias.*nansum(cond_ent_bias_nn,2)');  
    H_base_pred_nn(i)=-1*nansum(entr_u_pred_nn);
    CondH_base_pred_nn(i)=sum(p_s_base.*nansum(cond_ent_base_pred_nn,2)');
    H_bias_pred_nn(i)=-1*nansum(entr_b_pred_nn);
    CondH_bias_pred_nn(i)=sum(p_s_bias.*nansum(cond_ent_bias_pred_nn,2)');
    
    H_base_wn(i)=-1*nansum(entr_u_wn);                          
    CondH_base_wn(i)=sum(p_s_base.*nansum(cond_ent_base_wn,2)');
    H_bias_wn(i)=-1*nansum(entr_b_wn);
    CondH_bias_wn(i)=sum(p_s_bias.*nansum(cond_ent_bias_wn,2)');  
    H_base_pred_wn(i)=-1*nansum(entr_u_pred_wn);
    CondH_base_pred_wn(i)=sum(p_s_base.*nansum(cond_ent_base_pred_wn,2)');
    H_bias_pred_wn(i)=-1*nansum(entr_b_pred_wn);
    CondH_bias_pred_wn(i)=sum(p_s_bias.*nansum(cond_ent_bias_pred_wn,2)');
    
    H_base_ww(i)=-1*nansum(entr_u_ww);                          
    CondH_base_ww(i)=sum(p_s_base.*nansum(cond_ent_base_ww,2)');
    H_bias_ww(i)=-1*nansum(entr_b_ww);
    CondH_bias_ww(i)=sum(p_s_bias.*nansum(cond_ent_bias_ww,2)');  
    H_base_pred_ww(i)=-1*nansum(entr_u_pred_ww);
    CondH_base_pred_ww(i)=sum(p_s_base.*nansum(cond_ent_base_pred_ww,2)');
    H_bias_pred_ww(i)=-1*nansum(entr_b_pred_ww);
    CondH_bias_pred_ww(i)=sum(p_s_bias.*nansum(cond_ent_bias_pred_ww,2)');
%     stop
    if i<neurons
        clear tmp* bins* a* entr* cond* up_* bp_* b_xx* u_xx* xxu xx
    end
end
Ib_nw=H_bias_nw+CondH_bias_nw;                  % adapted TC w/ biased dist
Iu_nw=H_base_nw+CondH_base_nw;                  % unadapted TC w/ uniform dist
Iu_pred_nw=H_base_pred_nw+CondH_base_pred_nw;   % adapted TC w/ uniform dist
Ib_pred_nw=H_bias_pred_nw+CondH_bias_pred_nw;   % unadapted TC w/ biased dist

Ib_nn=H_bias_nn+CondH_bias_nn;
Iu_nn=H_base_nn+CondH_base_nn;
Iu_pred_nn=H_base_pred_nn+CondH_base_pred_nn;
Ib_pred_nn=H_bias_pred_nn+CondH_bias_pred_nn;

Ib_wn=H_bias_wn+CondH_bias_wn;
Iu_wn=H_base_wn+CondH_base_wn;
Iu_pred_wn=H_base_pred_wn+CondH_base_pred_wn;
Ib_pred_wn=H_bias_pred_wn+CondH_bias_pred_wn;

Ib_ww=H_bias_ww+CondH_bias_ww;
Iu_ww=H_base_ww+CondH_base_ww;
Iu_pred_ww=H_base_pred_ww+CondH_base_pred_ww;
Ib_pred_ww=H_bias_pred_ww+CondH_bias_pred_ww;

%% single unit entropy figures
figure
supertitle('MI single unit; n bw, w kern')
subplot(241)
plot(Iu_nw,Ib_nw,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('"data"')
xlabel('MI base')
ylabel('MI bias')
subplot(242)
plot(Ib_nw-Iu_nw,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(245)
plot(Iu_pred_nw,Ib_pred_nw,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:1),(0:0.1:1),':r')
title('prediction')
xlabel('MI base')
subplot(246)
plot(Ib_pred_nw-Iu_pred_nw,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
axis square;box off
subplot(243)
frac=100*(Ib_nw-Iu_nw)./Iu_nw;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(244)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(247)
frac_p=100*(Ib_pred_nw-Iu_pred_nw)./Iu_pred_nw;
frac_p(frac_p<-100)=-100;
frac_p(frac_p>100)=100;
bins=(-100:10:100);
a=histc(frac_p,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac_p),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(248)
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac_p(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')

figure
supertitle('MI single unit; n bw, n kern')
subplot(241)
plot(Iu_nn,Ib_nn,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('"data"')
xlabel('MI base')
ylabel('MI bias')
subplot(242)
plot(Ib_nn-Iu_nn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(245)
plot(Iu_pred_nn,Ib_pred_nn,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:1),(0:0.1:1),':r')
title('prediction')
xlabel('MI base')
subplot(246)
plot(Ib_pred_nn-Iu_pred_nn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
axis square;box off
subplot(243)
frac=100*(Ib_nn-Iu_nn)./Iu_nn;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(244)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(247)
frac_p=100*(Ib_pred_nn-Iu_pred_nn)./Iu_pred_nn;
frac_p(frac_p<-100)=-100;
frac_p(frac_p>100)=100;
bins=(-100:10:100);
a=histc(frac_p,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac_p),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(248)
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac_p(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')

figure
supertitle('MI single unit; w bw, n kern')
subplot(241)
plot(Iu_wn,Ib_wn,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('"data"')
xlabel('MI base')
ylabel('MI bias')
subplot(242)
plot(Ib_wn-Iu_wn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(245)
plot(Iu_pred_wn,Ib_pred_wn,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:1),(0:0.1:1),':r')
title('prediction')
xlabel('MI base')
subplot(246)
plot(Ib_pred_wn-Iu_pred_wn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
axis square;box off
subplot(243)
frac=100*(Ib_wn-Iu_wn)./Iu_wn;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(244)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(247)
frac_p=100*(Ib_pred_wn-Iu_pred_wn)./Iu_pred_wn;
frac_p(frac_p<-100)=-100;
frac_p(frac_p>100)=100;
bins=(-100:10:100);
a=histc(frac_p,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac_p),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(248)
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac_p(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')

figure
supertitle('MI single unit; w bw, w kern')
subplot(241)
plot(Iu_ww,Ib_ww,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('"data"')
xlabel('MI base')
ylabel('MI bias')
subplot(242)
plot(Ib_ww-Iu_ww,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(245)
plot(Iu_pred_ww,Ib_pred_ww,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:1),(0:0.1:1),':r')
title('prediction')
xlabel('MI base')
subplot(246)
plot(Ib_pred_ww-Iu_pred_ww,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
axis square;box off
subplot(243)
frac=100*(Ib_ww-Iu_ww)./Iu_ww;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(244)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(247)
frac_p=100*(Ib_pred_ww-Iu_pred_ww)./Iu_pred_ww;
frac_p(frac_p<-100)=-100;
frac_p(frac_p>100)=100;
bins=(-100:10:100);
a=histc(frac_p,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac_p),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(248)
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac_p(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac_p(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')

% % % % % % % % % biased TCs in uniform ensemble vs bias ensemble % % % % %
figure
supertitle('MI single unit; Effect of distribution')
subplot(4,4,1)
plot(Iu_pred_ww,Ib_ww,'.k')
axis([0 1 0 1])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('w bw, w kern')
xlabel('MI adapted TC w/ base')
ylabel('MI adapted TC w/ bias')
subplot(4,4,2)
plot(Ib_ww-Iu_pred_ww,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(4,4,3)
frac=100*(Ib_ww-Iu_pred_ww)./Iu_pred_ww;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(4,4,4)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(4,4,5)
plot(Iu_pred_wn,Ib_wn,'.k')
axis([0 0.5 0 0.5])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('w bw, n kern')
xlabel('MI adapted TC w/ base')
ylabel('MI adapted TC w/ bias')
subplot(4,4,6)
plot(Ib_wn-Iu_pred_wn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(4,4,7)
frac=100*(Ib_wn-Iu_pred_wn)./Iu_pred_wn;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(4,4,8)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(4,4,9)
plot(Iu_pred_nn,Ib_nn,'.k')
axis([0 0.5 0 0.5])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('n bw, n kern')
xlabel('MI adapted TC w/ base')
ylabel('MI adapted TC w/ bias')
subplot(4,4,10)
plot(Ib_nn-Iu_pred_nn,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(4,4,11)
frac=100*(Ib_nn-Iu_pred_nn)./Iu_pred_nn;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(4,4,12)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')
subplot(4,4,13)
plot(Iu_pred_nw,Ib_nw,'.k')
axis([0 0.5 0 0.5])
axis square;box off
hold on
plot((0:0.1:0.5),(0:0.1:0.5),':r')
title('w bw, w kern')
xlabel('MI adapted TC w/ base')
ylabel('MI adapted TC w/ bias')
subplot(4,4,14)
plot(Ib_nw-Iu_pred_nw,'.k')
hold on;box off
plot((1:70),zeros(70,1),':r')
xlabel('Oripref (deg)')
ylabel('MI_b_i_a_s-MI_b_a_s_e')
axis square;box off
subplot(4,4,15)
frac=100*(Ib_nw-Iu_pred_nw)./Iu_pred_nw;
bins=(-100:10:100);
a=histc(frac,bins);
bar(bins+diff(bins(1:2))/2,a/sum(a),1);
hold on; axis square; box off
plot(mean(frac),0.4,'vk')
set(gca,'PlotBoxAspectRatio',[3 2 1])
xlabel('Change in MI (%)')
ylabel('Proportion of cases')
subplot(4,4,16)
oripref_bins=[168.75 11.25:22.5:168.75];
for i=1:length(oripref_bins)-1
    if i==1
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i)));
        tmp2=frac(find(ori_prefs(1:70)<oripref_bins(i+1)));
        tmp=union(tmp,tmp2);
    else
        tmp=frac(find(ori_prefs(1:70)>oripref_bins(i) & ori_prefs(1:70)<oripref_bins(i+1)));
    end
    tmp(tmp<-100)=-100;
    tmp(tmp>100)=100;
    dmi(i)=mean(tmp);
    dmi_se(i)=std(tmp)/length(tmp);
end
plotbins=(-90:22.5:90);
errorline(plotbins(1:end-1),circshift(dmi,[0 0]),circshift(dmi_se,[0 0]),'k')
hold on; axis square; box off
plot((-90:90),zeros(181,1),':k')
xlabel('Oripref (deg)')
set(gca,'PlotBoxAspectRatio',[3 2 1],'XLim',[-95 95],'XTick',[-90 -45 0 45 90])
ylabel('% change in MI')
xlabel('Orientation preference (deg)')

% SUMI doesn't seem to depend on # of trials or kernel.
% shape doesn't much depend on amplitude of TC alone but
% if bandwidth is narrow and amplitude high, bias preferring neurons lose info.

%% look at entropy as a function 1)TC bandwidth, 2) orientation pref, 3) trials, 4) bias ratio
clear

% set trials, pop size, and ori prefs/oris used
trials=[50 100 300 500 1000 2000 4000 8000];       % approximate avg of trials in awake data
neurons=3;         % approximate avg of # of units in awake data
% trial_match=300;    % approximate avg of trials used in neighbors classification in awake data
oris_coarse=10:20:170;   % oris used in experiment
% oris_fine=0:2.5:177.5;  % oris for neighbors classification
ori_prefs=[10 50 90];

bias=[10 14 18 23]; % 2:1, 6:1, 10:1, 15:1
widths=[0 3 6];

% create tuning curves: adam's way
ampl=11*ones(neurons,1);%*rand(neurons,1);
base=1*ones(neurons,1);%*rand(neurons,1);
tmp=rand(neurons,1)-0.5;

for r = 1:length(trials)
    for q=1:length(bias)
        for p=1:length(widths)
            
            bw=(widths(p)+4)*ones(neurons,1);%*tmp; % originally 3.5 + 8
%             bw=widdths(p)+4*tmp
            bw(bw<0)=0.01;
            
            % wide kernel:  using strong (0.5) kernel
            kernelw=exp(2*(cos(2*deg2rad(oris_coarse-90))-1)); % kernel centered at 90
            kernelw=1-0.5*kernelw;
            
            for i=1:neurons
                aktune(i,:)=base(i)+ampl(i)*exp(bw(i)*(cos(2*deg2rad(oris_coarse-ori_prefs(i)))-1));
                aktune_a(i,:)=aktune(i,:).*kernelw;
            end
            
            u_dist=randi(9,trials(r),1);
            b_dist=randi(bias(q),trials(r),1); % 14
            b_dist(b_dist>9)=5;
            
            for i=1:neurons
                resp_uu(i,:)=poissrnd(aktune(i,u_dist));
                resp_ab(i,:)=poissrnd(aktune_a(i,b_dist));
                resp_ub(i,:)=poissrnd(aktune(i,b_dist));
                resp_au(i,:)=poissrnd(aktune_a(i,u_dist));
                
                bin=0:1:max([(resp_uu(i,:)) (resp_ub(i,:))...
                    (resp_au(i,:)) squeeze(resp_ab(i,:))]);
                
                a1=histc((resp_uu(i,:)),bin)/trials(r);
                a2=histc((resp_ab(i,:)),bin)/trials(r);
                a4=histc((resp_ub(i,:)),bin)/trials(r);
                a3=histc((resp_au(i,:)),bin)/trials(r);
                
                for j=1:length(bin)
                    entr_u(j)=a1(j)*log2(a1(j));
                    entr_b(j)=a2(j)*log2(a2(j));
                    entr_u_pred(j)=a4(j)*log2(a4(j));
                    entr_b_pred(j)=a3(j)*log2(a3(j));
                end
                
                for j=1:length(oris_coarse)
                    xxu=find(u_dist==j);     % cases of each ori
                    u_xx=histc(resp_uu(i,xxu),bin)/length(xxu);    % P(r) in each bin for given ori
                    bp_xx=histc(resp_au(i,xxu),bin)/length(xxu);
                    
                    p_s_base(j)=length(xxu)/length(u_dist);  % probability of each stimulus in uniform dist
                    for k=1:length(bin)
                        cond_ent_base(j,k)=u_xx(k)*log2(u_xx(k));       % conditional entropy for uniform distribution
                        cond_ent_bias_pred(j,k)=bp_xx(k)*log2(bp_xx(k));% cond. entropy for bias TC in uniform distribution
                    end
                    
                    xx=find(b_dist==j);
                    b_xx=histc(resp_ab(i,xx),bin)/length(xx);    % P(r) in each bin for given ori
                    up_xx=histc(resp_ub(i,xx),bin)/length(xx);
                    p_s_bias(j)=length(xx)/length(b_dist);  % probability of each stimulus in bias dist
                    for k=1:length(bin)
                        cond_ent_bias(j,k)=b_xx(k)*log2(b_xx(k));   % conditional entropy for uniform
                        cond_ent_base_pred(j,k)=up_xx(k)*log2(up_xx(k)); %cond. entropy for bias pred
                    end
                    clear xxu u_xx bp_xx xx b_xx 
                end
                H_base(i)=-1*nansum(entr_u);              % unadapted TC, uniform dist
                CondH_base(i)=sum(p_s_base.*nansum(cond_ent_base,2)');
                H_bias(i)=-1*nansum(entr_b);              % adapted TC, bias dist
                CondH_bias(i)=sum(p_s_bias.*nansum(cond_ent_bias,2)');
                H_base_pred(i)=-1*nansum(entr_u_pred);    % Adapted TC, uniform dist
                CondH_base_pred(i)=sum(p_s_base.*nansum(cond_ent_base_pred,2)');
                H_bias_pred(i)=-1*nansum(entr_b_pred);    % unadapted TC, bias dist
                CondH_bias_pred(i)=sum(p_s_bias.*nansum(cond_ent_bias_pred,2)');
            end
            
            Ib(r,q,p,:)=H_bias+CondH_bias;                  % adapted TC w/ biased dist
            Iu(r,q,p,:)=H_base+CondH_base;                  % unadapted TC w/ uniform dist
            Iu_pred(r,q,p,:)=H_base_pred+CondH_base_pred;   % adapted TC w/ uniform dist
            Ib_pred(r,q,p,:)=H_bias_pred+CondH_bias_pred;   % unadapted TC w/ biased dist
        end
        clear resp*
    end
end
% trials x bias x TCwidth x neurons

figure
supertitle('trials vs bias (wide BW)')
subplot(231)
imagesc(squeeze(Iu(:,:,1,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, null pref')
set(gca,'YDir','normal','XTickLabel',{'2:1','6:1','10:1','15:1'},'YTick',1:8,'YTickLabel',{'50','100','300','500','1k','2k','4k','8k'})
subplot(232)
imagesc(squeeze(Iu(:,:,1,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, offset pref')
set(gca,'YDir','normal')
subplot(233)
imagesc(squeeze(Iu(:,:,1,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, adapt pref')
set(gca,'YDir','normal')
subplot(234)
imagesc(squeeze(Ib(:,:,1,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, null pref')
set(gca,'YDir','normal')
subplot(235)
imagesc(squeeze(Ib(:,:,1,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, offset pref')
set(gca,'YDir','normal')
subplot(236)
imagesc(squeeze(Ib(:,:,1,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, adapt pref')
set(gca,'YDir','normal')

figure
supertitle('trials vs bias (mid BW)')
subplot(231)
imagesc(squeeze(Iu(:,:,2,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, null pref')
set(gca,'YDir','normal','XTickLabel',{'2:1','6:1','10:1','15:1'},'YTick',1:8,'YTickLabel',{'50','100','300','500','1k','2k','4k','8k'})
subplot(232)
imagesc(squeeze(Iu(:,:,2,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, offset pref')
set(gca,'YDir','normal')
subplot(233)
imagesc(squeeze(Iu(:,:,2,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, adapt pref')
set(gca,'YDir','normal')
subplot(234)
imagesc(squeeze(Ib(:,:,2,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, null pref')
set(gca,'YDir','normal')
subplot(235)
imagesc(squeeze(Ib(:,:,2,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, offset pref')
set(gca,'YDir','normal')
subplot(236)
imagesc(squeeze(Ib(:,:,2,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, adapt pref')
set(gca,'YDir','normal')

figure
supertitle('trials vs bias (narrow BW)')
subplot(231)
imagesc(squeeze(Iu(:,:,3,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, null pref')
set(gca,'YDir','normal','XTickLabel',{'2:1','6:1','10:1','15:1'},'YTick',1:8,'YTickLabel',{'50','100','300','500','1k','2k','4k','8k'})
subplot(232)
imagesc(squeeze(Iu(:,:,3,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, offset pref')
set(gca,'YDir','normal')
subplot(233)
imagesc(squeeze(Iu(:,:,3,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Uniform, adapt pref')
set(gca,'YDir','normal')
subplot(234)
imagesc(squeeze(Ib(:,:,3,1)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, null pref')
set(gca,'YDir','normal')
subplot(235)
imagesc(squeeze(Ib(:,:,3,2)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, offset pref')
set(gca,'YDir','normal')
subplot(236)
imagesc(squeeze(Ib(:,:,3,3)),[0 1])
box off, axis square
ylabel('Trials'); xlabel('Bias'); title('Bias, adapt pref')
set(gca,'YDir','normal')