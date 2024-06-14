% 2023-12-26 wisdom 
% Water level predict results anlysis

clc;
close all;
clear all;

% 数据导入
load ('Result_15D.mat');   % 19个站点15天水位预测结果

%% 预测结果展示
for DayID=15
    figure; hold on;
%     Result_15D(DayID,3:7)=Result_15D(DayID,3:7)-150;
%     Result_15D(DayID+15,3:7)=Result_15D(DayID+15,3:7)-150;
    plot(Result_15D(DayID,:),'-b*'); %观测值
    hold on;
    plot(Result_15D(DayID+15,:),'-rs');%预测值
    legend('Measured','Predicted');
%     title([num2str(DayID)  'Day']);
    set(gca,'XLim',[0 20]);
    set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
    set(gca,'FontSize',22);  %是设置刻度字体大小
    ylabel('Water level (m)');
    xlabel('Stations');
    box on;
    grid on;
    hold off;
 end 

%% 代表性站点水位展示
for StaID=12
    figure; hold on;
    plot(Result_15D(1:15,StaID),'-bs'); %观测值
    hold on;
    plot(Result_15D(16:30,StaID),'-r^');%预测值
    legend('Measured','Predicted');
    set(gca,'XLim',[0 16]);
    set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
    set(gca,'FontSize',20);  %是设置刻度字体大小
    ylabel('Water level (m)');
    xlabel('Days');
    box on;
    grid on;
    hold off;
end
    %----1站
    figure; 
    hold on;
    subplot(3, 1, 1);
    plot(Result_15D(1:15,1),'-bs'); %观测值
    hold on;
    plot(Result_15D(16:30,1),'-r^');%预测值
    hold on;
    set(gca,'XLim',[0 16]);
    legend('Measured','Predicted');
    set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
    set(gca,'FontSize',14);  %是设置刻度字体大小
    ylabel('Water level (m)');
    xlabel('Days');
    title('Station 1: Yibin');
    box on;
    grid on;
    hold off;
    %----3站
    subplot(3, 1, 2);
    plot(Result_15D(1:15,12),'-bs'); %观测值
    hold on;
    plot(Result_15D(16:30,12),'-r^');%预测值
    hold on;
    set(gca,'XLim',[0 16]);
    legend('Measured','Predicted');
    set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
    set(gca,'FontSize',14);  %是设置刻度字体大小
    ylabel('Water level (m)');
    xlabel('Days');
    title('Station 12: Chenglingji');
    box on;
    grid on;
    hold off;
    %----19站
    subplot(3, 1, 3);
    plot(Result_15D(1:15,19),'-bs'); %观测值
    hold on;
    plot(Result_15D(16:30,19),'-r^');%预测值
    set(gca,'XLim',[0 16]);
    legend('Measured','Predicted');
    set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
    set(gca,'FontSize',14);  %是设置刻度字体大小
    ylabel('Water level (m)');
    xlabel('Days');
    title('Station 19: Zhenjiang');
    box on;
    grid on;
    hold off;

%% 19个站点水位预测误差分析：NSE RMSE MAPE MAE
for StaID=1:19
    %--计算19个站点1-15每天的AE、APE
    AE(:,StaID)=abs(Result_15D(1:15,StaID)-Result_15D(16:30,StaID));
    APE(:,StaID)=abs((Result_15D(1:15,StaID)-Result_15D(16:30,StaID))./Result_15D(1:15,StaID))*100;
    for DayID=1:15
        MeaWater=Result_15D(1:DayID,StaID);
        PreWater=Result_15D(16:DayID+15,StaID);
        % -----RMSE----
        RMSE19(DayID,StaID)=sqrt(sum((MeaWater-PreWater).^2)/length(MeaWater));
        % -----MAPE----
        MAPE19(DayID,StaID)=sum(abs(MeaWater-PreWater)./abs(MeaWater))/length(MeaWater)*100;
        % -----MAE----
        MAE19(DayID,StaID)=sum(abs(MeaWater-PreWater))/length(MeaWater);
    end
end
% StaID=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19];
figure; hold on;
plot(AE(1,:),'-b*');  %第1天
hold on;
plot(AE(3,:),'-ks');  %第3天
hold on;
plot(AE(5,:),'-gv');  %第5天
hold on;
plot(AE(7,:),'-md');  %第7天
hold on;
plot(AE(10,:),'-cx'); %第10天
hold on;
plot(AE(15,:),'-r^'); %第15天
% hold on;
% plot(AE(7,:),'-r^');
legend('Future 1st day','Future 3rd day','Future 5th day','Future 7th day','Future 10th day','Future 15th day');
set(gca,'XLim',[0 20]);
set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
set(gca,'FontSize',20);  %是设置刻度字体大小
ylabel('AE (m)');
xlabel('Stations');
box on;
grid on;
hold off; 

figure; hold on;
plot(APE(1,:),'-b*');  %第1天
hold on;
plot(APE(3,:),'-ks');  %第3天
hold on;
plot(APE(5,:),'-gv');  %第5天
hold on;
plot(APE(7,:),'-md');  %第7天
hold on;
plot(APE(10,:),'-cx'); %第10天
hold on;
plot(APE(15,:),'-r^'); %第15天
% hold on;
% plot(AE(7,:),'-ko');
legend('Future 1st day','Future 3rd day','Future 5th day','Future 7th day','Future 10th day','Future 15th day');
set(gca,'XLim',[0 20]);
set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
set(gca,'FontSize',20);  %是设置刻度字体大小
ylabel('APE (%)');
xlabel('Stations');
box on;
grid on;
hold off; 

figure; hold on;
plot(MAPE19(1,:),'-b*');  %第1天
hold on;
plot(MAPE19(3,:),'-ks');  %第3天
hold on;
plot(MAPE19(5,:),'-gv');  %第5天
hold on;
plot(MAPE19(7,:),'-md');  %第7天
hold on;
plot(MAPE19(10,:),'-cx'); %第10天
hold on;
plot(MAPE19(15,:),'-r^'); %第15天
% hold on;
% plot(AE(7,:),'-ko');
legend('Future 1st day','Future 3rd day','Future 5th day','Future 7th day','Future 10th day','Future 15th day');
set(gca,'XLim',[0 20]);
set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
set(gca,'FontSize',20);  %是设置刻度字体大小
ylabel('MAPE (%)');
xlabel('Stations');
box on;
grid on;
hold off; 

% 19个站点15天的AE
figure; hold on;
for StaID=1:15
   plot(AE(StaID,:),'MarkerSize',15');
   hold on; 
end
set(gca,'XLim',[0 20]);
set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
set(gca,'FontSize',20);  %是设置刻度字体大小
legend('Day 1','Day 2','Day 3','Day 4','Day 5','Day 6','Day 7','Day 8','Day 9','Day 10','Day 11','Day 12','Day 13','Day 14','Day 15');
ylabel('AE (m)');
xlabel('Stations');
box on;
grid on;
hold off; 
%% 15天水位预测误差分析：NSE 
MeaWater=[];
PreWater=[];
for DayID=1:15
    MeaTemp=Result_15D(DayID,:);
    PreTemp=Result_15D(DayID+15,:);
%     %每一天的NSE
%     NSE152(DayID)=1-(sum(power(MeaTemp-PreTemp,2))/sum(power(MeaTemp-mean(MeaTemp),2)));
    
    MeaWater=[MeaWater,MeaTemp];
    PreWater=[PreWater,PreTemp];
    %累积的NSE
    NSE15(DayID)=1-(sum((MeaWater-PreWater).^2)/sum((MeaWater-mean(MeaWater)).^2));
    % -----RMSE----
    RMSE15(DayID)=sqrt(sum((MeaWater-PreWater).^2)/length(MeaWater));
    % -----MAPE----
    MAPE15(DayID)=sum(abs(MeaWater-PreWater)./abs(MeaWater))/length(MeaWater)*100;
    % -----MAE----
    MAE15(DayID)=sum(abs(MeaWater-PreWater))/length(MeaWater);
end

% Errors=[MAE15;RMSE15;NSE15];
% figure; hold on;
% bar(Errors');
% legend('MAE','RMSE','NSE');
% set(gca,'XLim',[0 16]);
% set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
% set(gca,'FontSize',16);  %是设置刻度字体大小
% ylabel('Prediction Errors');
% xlabel('Days');
% box on;
% grid on;
% hold off; 

figure; hold on;
plot(MAE15,'-bs');  
hold on;
plot(MAPE15/100,'-m*'); 
hold on;
plot(RMSE15,'-g^');  
hold on;
plot(NSE15,'-r*');  
legend('MAE','MAPE','RMSE','NSE');
set(gca,'XLim',[0 16]);
set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
set(gca,'FontSize',16);  %是设置刻度字体大小
ylabel('Prediction Errors');
xlabel('Days');
box on;
grid on;
hold off; 

%% 19个站点15天的预测水位

% ----Stations 1-7----------------
% hold on;
% plot(Result_15D(16,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(17,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(18,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(19,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(20,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(21,:),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(22,:),'MarkerSize',10');  %
% legend('Day 1','Day 2','Day 3','Day 4','Day 5','Day 6','Day 7');

% ----Stations 8-10+17-19----------------
% hold on;
% plot(Result_15D(16:30,8),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,9)+5,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,10)+8,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,17)-6,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,18)-8,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,19)-10,'MarkerSize',10');  %
% legend('Station 8','Station 9','Station 10','Station 17','Station 18','Station 19');

% ----Stations 11-16------------
% hold on;
% plot(Result_15D(16:30,11)-2,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,12),'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,13)+3.5,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,14)+7,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,15)-7,'MarkerSize',10');  %
% hold on;
% plot(Result_15D(16:30,16)-10,'MarkerSize',10');  %
% legend('Station 11','Station 12','Station 13','Station 14','Station 15','Station 16');

% set(gca,'FontName','Times New Roman'); %是设置刻度字体型号
% set(gca,'FontSize',19);  %是设置刻度字体大小
% set(gca,'XLim',[0 16]);
% % set(gca,'Xtick',[1 182 366 547 731 913 1096 1277 1461 1643],'fontsize',18);
% % set(gca,'Xticklabel',{'2018-Jan','Jun','2019-Jan','Jun','2020-Jan','Jun','2021-Jan','Jun','2022-Jan','Jun'});
% ylabel('Day');
% ylabel('Water level (m)');
% box on;
% hold off;

%% 水运能耗数据
IsFull=[12.8 10.95 44.02 2.45 1.53 3.08];
subplot(1,2,1);
bar([12.8 10.95 44.02]); % 示例图形
 
subplot(1,2,2);
bar([2.45 1.53 3.08]); % 另一个示例图形
box on;
hold off;