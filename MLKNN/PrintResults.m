
function PrintResults(Result)
    [~,n] = size(Result);
    if n == 2
        fprintf('------------------------------------\n');
        fprintf('Evalucation Metric    Mean    Std\n');
        fprintf('------------------------------------\n');
        fprintf('HammingLoss           %.4f  %.4f\r',Result(1,1),Result(1,2));
        fprintf('ExampleBasedAccuracy  %.4f  %.4f\r',Result(2,1),Result(2,2));
        fprintf('ExampleBasedPrecision %.4f  %.4f\r',Result(3,1),Result(3,2));
        fprintf('ExampleBasedRecall    %.4f  %.4f\r',Result(4,1),Result(4,2));
        fprintf('ExampleBasedFmeasure  %.4f  %.4f\r',Result(5,1),Result(5,2));

        fprintf('SubsetAccuracy        %.4f  %.4f\r',Result(6,1),Result(6,2));
        fprintf('LabelBasedAccuracy    %.4f  %.4f\r',Result(7,1),Result(7,2));
        fprintf('LabelBasedPrecision   %.4f  %.4f\r',Result(8,1),Result(8,2));
        fprintf('LabelBasedRecall      %.4f  %.4f\r',Result(9,1),Result(9,2));
        fprintf('LabelBasedFmeasure    %.4f  %.4f\r',Result(10,1),Result(10,2));

        fprintf('MicroF1Measure        %.4f  %.4f\r',Result(11,1),Result(11,2));
        fprintf('Average_Precision     %.4f  %.4f\r',Result(12,1),Result(12,2));
        fprintf('OneError              %.4f  %.4f\r',Result(13,1),Result(13,2));
        fprintf('RankingLoss           %.4f  %.4f\r',Result(14,1),Result(14,2));
        fprintf('Coverage              %.4f  %.4f\r',Result(15,1),Result(15,2));
        fprintf('------------------------------------\n');
    else
        fprintf('\n----------------------------\n');
        fprintf('Evalucation Metric    Mean\n');
        fprintf('----------------------------\n');
        fprintf('HammingLoss           %.4f\r',Result(1,1));
        fprintf('ExampleBasedAccuracy  %.4f\r',Result(2,1));
        fprintf('ExampleBasedPrecision %.4f\r',Result(3,1));
        fprintf('ExampleBasedRecall    %.4f\r',Result(4,1));
        fprintf('ExampleBasedFmeasure  %.4f\r',Result(5,1));

        fprintf('SubsetAccuracy        %.4f\r',Result(6,1));
        fprintf('LabelBasedAccuracy    %.4f\r',Result(7,1));
        fprintf('LabelBasedPrecision   %.4f\r',Result(8,1));
        fprintf('LabelBasedRecall      %.4f\r',Result(9,1));
        fprintf('LabelBasedFmeasure    %.4f\r',Result(10,1));

        fprintf('MicroF1Measure        %.4f\r',Result(11,1));
        fprintf('Average_Precision     %.4f\r',Result(12,1));
        fprintf('OneError              %.4f\r',Result(13,1));
        fprintf('RankingLoss           %.4f\r',Result(14,1));
        fprintf('Coverage              %.4f\r',Result(15,1));
        fprintf('----------------------------\n');
    end
end