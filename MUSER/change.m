load("Mirflickr");
% data = double(cat(2, data{:}));
data = double(l);
target = double(target);
save Mirflickr.mat data target;