function [STATS TX_OK X m S posto]=mahalanobis(data,Nr,Ptrain)
%
% Mahalanobis with one COV matrix per class.
%
% INPUTS: * data (matrix): dataset matrix (N x (p+1))
%	  	OBS1: feature vectors along the rows of data matrix
%	  	OBS2: last column must contain numerical class labels
%	  * Nr (scalar): Number of runs (Nr>=1)
%	  * Ptrain (scalar): Percentage of training data (0 < Ptrain < 100)
%
% OUTPUTS: X (struct) - the data samples separated per class
%          m (struct) - the classes centroids
%          S (struct) - the COV matrices per class
%          STATS (vector) - Statistics of test data (mean, median, min/max, sd)
%
% Author: Guilherme Barreto
% Date: 26/07/2025

[N p]=size(data);  % Get dataset size (N)

Ntrn=round(Ptrain*N/100);  % Number of training samples
Ntst=N-Ntrn; % Number of testing samples

K=max(data(:,end)); % Get the number of classes

ZZ=sprintf('The problem has %d classes',K);
disp(ZZ);

for r=1:Nr,  % Loop of independent runs

  I=randperm(N);
  data=data(I,:); % Shuffle rows of the data matrix
  %data(:,1:end-1)=data(:,1:end-1)+0.5*randn(size(data(:,1:end-1)));
  
  % Separate into training and testing subsets
  Dtrn=data(1:Ntrn,:);  % Training data
  Dtst=data(Ntrn+1:N,:); % Testing data

  % Partition of training data into K subsets
  for k=1:K,
    I=find(Dtrn(:,end)==k);  % Find rows with samples from k-th class
    X{k}=Dtrn(I,1:end-1); % Data samples from k-th class
    m{k}=mean(X{k})';   % Centroid of the k-th class
    S{k}=cov(X{k}); % Compute the covariance matrix of the k-th class
    posto{k}=rank(S{k}); % Check invertibility of covariance matrix by its rank
    %iS{k}=pinv(S{k});    % Inverse covariance matrix of the k-th class
    iS{k}=inv(S{k});    % Inverse covariance matrix of the k-th class
  end

  % Testing phase
  correct=0;  % number correct classifications
  for i=1:Ntst,
    Xtst=Dtst(i,1:end-1)';   % test sample to be classified
    Label_Xtst=Dtst(i,end);   % Actual label of the test sample
    for k=1:K,
      v=(Xtst-m{k});
      dist(k)=v'*iS{k}*v + log(det(S{k}));  % Mahalanobis distance to k-th class
      %pause
    end
    [dummy Pred_class]=min(dist);  % index of the minimum distance class
    
    if Pred_class == Label_Xtst,
        correct=correct+1;
    end
  end
  
  TX_OK(r)=100*correct/Ntst;   % Recognition rate of r-th run
end

STATS=[mean(TX_OK) min(TX_OK) max(TX_OK) median(TX_OK) std(TX_OK)];
