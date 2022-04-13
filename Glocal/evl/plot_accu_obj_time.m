close all;
l1 = numel(find(out1.top1));
l2 = numel(find(out2.top1));
l3 = numel(find(out3.top1));
%%%%%%%%%%%plot top-1 accu%%%%%%%%%%%
figure;
plot(out1.time(1:l1), out1.top1(1:l1), '--bs', 'LineWidth', 1);
hold on
plot(out2.time(1:l2), out2.top1(1:l2), '--ro','LineWidth',1);
plot(out3.time(1:l3), out3.top1(1:l3), '--go','LineWidth',1);
 xlabel('time (s)');
ylabel('Top1 Accuracy');
title('Time VS top-1 Accu');
legend('R1MP','LEML');
hold off

%%%%%%%%%%%%%plot top-3 accu%%%%%%%%%%%%
figure;
plot(out1.time(1:l1), out1.top3(1:l1), '--bs', 'LineWidth', 1);
hold on
plot(out2.time(1:l2), out2.top3(1:l2), '--ro','LineWidth',1);
plot(out3.time(1:l3), out3.top3(1:l3), '--go','LineWidth',1);
 xlabel('time (s)');
ylabel('Top3 Accuracy');
title('Time VS top-3 Accu');
legend('R1MP','LEML');
hold off

%%%%%%%%%%%%%%%plot obj %%%%%%%%%%%%%%
figure;
plot(out1.time(1:l1), out1.obj(1:l1), '--bs', 'LineWidth', 1);
hold on
plot(out2.time(1:l2), out2.obj(1:l2), '--ro','LineWidth',1);
plot(out3.time(1:l3), out3.obj(1:l3), '--go','LineWidth',1);
 xlabel('time (s)');
ylabel('obj');
title('Time VS obj');
legend('R1MP','LEML');
hold off


%%%%%%%%%%%%%%%plot log obj%%%%%%%%%%%%
figure;
plot(out1.time(1:l1), log(out1.obj(1:l1)), '--bs', 'LineWidth', 1);
hold on
plot(out2.time(1:l2), log(out2.obj(1:l2)), '--ro','LineWidth',1);
 xlabel('time (s)');
ylabel('logarithm of obj');
title('Time VS logarithm of obj');
legend('R1MP','LEML');
hold off
