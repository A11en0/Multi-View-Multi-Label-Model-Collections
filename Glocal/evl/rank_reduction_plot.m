rrrtop1 = out1.top1;
rrrtop3 = out1.top3;
for i = 1:len
    lemltop1(i) = max(out2(i).top1);
    lemltop3(i) = max(out2(i).top3);
    rankonetop1(i) = rrrtop1(i*step);
    rankonetop3(i)  = rrrtop3(i*step);
end

figure;
plot(step:step:len*step, rankonetop1,'--bs');
hold on;
plot(step:step:len*step, lemltop1,'--ro');
xlabel('rank');
ylabel('Top1 Accuracy');
title('corel5k - full label');
legend('R1MP','LEML');
hold off;



figure;
plot(step:step:len*step, rankonetop3,'--bs');
hold on;
plot(step:step:len*step, lemltop3,'--ro');
xlabel('rank');
ylabel('Top3 Accuracy');
title('corel5k - full label');
legend('R1MP','LEML');
hold off;


