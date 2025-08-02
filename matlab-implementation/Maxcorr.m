% Classificador de Maxima Correlacao (com Centroide calculado pela media)

% Autor: Guilherme Barreto
% Data (ultima atualizacao): 01/05/2025

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
C=max(Y);  % Num classes

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
  %%%% PROJETO DO CLASSIFICADOR DMC
  %%%%%%%%%%%

  Xtrn=X(:,1:Ntrn);  % Dados de treino: Coluna 1 => Coluna Ntrn
  Ytrn=Y(:,1:Ntrn);  % Rotulos dos vetores de treino

  for k=1:C,
    I=find(Ytrn==k);   % Encontra colunas cujos vetores sao da classe k
    M{k}=mean(Xtrn(:,I),2);   % Calcula o centroide do subconjunto de vetores da classe k
    LABEL{k}=k;   % Rotula o centroide correspondente
  endfor

  %%%%%%%%%%%
  %%%% TESTE DO CLASSIFICADOR DMC
  %%%%%%%%%%%

  Xtst=X(:,Ntrn+1:end);   % Dados de teste: Coluna Ntrn+1 => Ultima coluna
  Ytst=Y(:,Ntrn+1:end);  % Rotulos dos vetores de teste

  Ntst=N-Ntrn;   % Numero de dados de teste

  acerto=0;  % variavel que contabiliza os acertos
  for i=1:Ntst,  % loop para teste
    Xnew=Xtst(:,i);  % vetor de teste
    Ynew=Ytst(i);  % rotulo correto do vetor de teste

    for k=1:C,  % loop para varrer os exemplos de treino armazenados

       % Metodo 1: Usando os comandos DOT e NORM nativos do Octave/Matlab
       Sim(k) = dot(Xnew/norm(Xnew), M{k}/norm(M{k}));

       % Metodo 2: Usando o produto escalar (sem aplicar a raiz quadrada)
       %aux1 = Xnew/(Xnew'*Xnew);
       %aux2 = M{k}/(M{k}'*M{k});
       %Sim(k) = (Xnew/(Xnew'*Xnew))' * M{k}/(M{k}'*M{k});

    endfor

    [Smax Jmax]=max(Sim);   % Jmax = indice da coluna onde esta o vetor de treino mais similar

    if Jmax == Ynew,  % contabiliza acerto na classificacao
      acerto=acerto+1;
    endif
  endfor

  Pacerto(r)=100*acerto/Ntst;

end   % Fim do loop de repeticoes

t_maxcorr = toc

STATS=[mean(Pacerto) std(Pacerto) median(Pacerto) min(Pacerto) max(Pacerto)]
