% Classificador K-NN

% Autor: Guilherme Barreto
% Data: 07/04/2022

clear; clc; close all;

%X=load('atributosYALE.dat'); Y=load('rotulosYALE.dat');  % Conjunto Yale A
%X=load('atributosIRIS.dat'); Y=load('rotulosIRIS.dat');  % Conjunto IRIS
%X=load('atributosDERMO.dat'); Y=load('rotulosDERMO.dat');  % Conjunto Dermatology
X=load('atributosPARK.dat'); Y=load('rotulosPARK.dat');  % Conjunto Parkinson

%%%% NORMALIZACAO VARIANCIA UNITARIA
med=mean(X,2);   % Media dos atributos
dp=std(X,[],2);  % desvio padrao dos atributos
X=(X-med)./dp;   % Normaliza atributos  (Com DMT melhora, Com IRIS piora, Com YA fica no mesmo)

N=length(Y);   % Total de exemplos disponiveis

Ptrn=0.8;  % Porcentagem de dados de treinamento
Ntrn=floor(Ptrn*N);

Nrep=50; % Num. de repeticoes/realizacoes/rodadas de treino-teste

tic;

for r=1:Nrep,   % inicio do loop de repeticoes

  rodada=r,

  I=randperm(N);  % Embaralha indices das colunas de X
  X=X(:,I);  % Embaralha as colunas de X
  Y=Y(:,I);  % Leva os rotulos para as mesmas posicoes

  %%%%%%%%%%%
  %%%% PROJETO DO CLASSIFICADOR K-NN
  %%%%%%%%%%%

  Xtrn=X(:,1:Ntrn);  % Dados de treino: Coluna 1 => Coluna Ntrn
  Ytrn=Y(:,1:Ntrn);  % Rotulos dos vetores de treino

  %%%%%%%%%%%
  %%%% TESTE DO CLASSIFICADOR K-NN
  %%%%%%%%%%%

  Xtst=X(:,Ntrn+1:end);   % Dados de teste: Coluna Ntrn+1 => Ultima coluna
  Ytst=Y(:,Ntrn+1:end);  % Rotulos dos vetores de teste

  Ntst=N-Ntrn;   % Numero de dados de teste

  acerto=0;  % variavel que contabiliza os acertos
  for i=1:Ntst,  % loop para teste
    Xnew=Xtst(:,i);  % vetor de teste
    Ynew=Ytst(i);  % rotulo correto do vetor de teste

    for j=1:Ntrn,  % loop para varrer os exemplos de treino armazenados

      % Metodo 1: Usando o comando NORM nativo do Octave/Matlab
      %dists(j)=norm(Xnew-Xtrn(:,j));

      % Metodo 2: Usando o produto escalar (sem aplicar a raiz quadrada)
      aux=Xnew-Xtrn(:,j);
      dists(j)=aux'*aux;
    endfor

    [Dmin Jmin]=min(dists);   % Jmin = indice da coluna onde esta o vetor de treino mais proximo

    if Ytrn(Jmin) == Ynew,  % contabiliza acerto na classificacao
      acerto=acerto+1;
    endif
  endfor

  Pacerto(r)=100*acerto/Ntst;

end   % Fim do loop de repeticoes

t_knn = toc

STATS=[mean(Pacerto) std(Pacerto) median(Pacerto) min(Pacerto) max(Pacerto)]
