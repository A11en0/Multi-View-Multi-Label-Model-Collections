function [Vsort,Dsort] = eigsort(V,D);
eigvals = diag(D);
% Sort the eigenvalues from largest to smallest. Store the sorted
% eigenvalues in the column vector lambda.
[lohival,lohiindex] = sort(eigvals);
lambda = flipud(lohival);
index = flipud(lohiindex);
Dsort = diag(lambda);
% Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
% eigenvectors as columns of the matrix vsort.
M = length(lambda);
Vsort = zeros(M,M);
for i=1:M
  Vsort(:,i) = V(:,index(i));
end;

end
