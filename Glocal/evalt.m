function [ Result ] = evalt(Fpred,Ygnd,thr)
Ypred = sign(Fpred-thr);

%% Coverage
Cvg = coverage(Fpred,Ygnd);
disp(['Coverage: ',num2str(Cvg)]);
Result.Coverage = Cvg;

%% Average Precision
AvgPrec = Average_precision(Fpred,Ygnd);
disp(['Average Precision: ',num2str(AvgPrec)]);
Result.AveragePrecision = AvgPrec;

%% Ranking Loss
RkL = Ranking_loss(Fpred,Ygnd);
disp(['Ranking Loss: ',num2str(RkL)]);
Result.RankingLoss = RkL;

Result.Top1 = topRate(Fpred',Ygnd',1);
Result.Top3 = topRate(Fpred',Ygnd',3);
Result.Top5 = topRate(Fpred',Ygnd',5);

Result.AvgAuc = avgauc(Fpred,Ygnd);