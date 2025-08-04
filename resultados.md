## Resultados observados

### Tabela 1

O modelo de normalização que apresentou melhores resultados foi o sem normalização, portanto a tabela 1 fica da seguinte maneira:

![alt text](image-3.png)

### Questão 1

Os classificadores baseados na matriz de covariância apresentaram desempenhos comprometidos, provavelmente devido ao grande número de atributos e poucos dados. Já os classificadores que não se baseiam nela, tiveram desempenhos melhores. 
### Questão 2

- Em relação à taxa de acerto: `Variante de Tikhonov`
- Em relação ao tempo: `MaxCorr`

### Questão 3

Sim, para o classificador quadratico default e para a variante 4 (Naive Bayes), o qual usa a matriz Ci,diag assumindo que os atributos são
descorrelacionados para aquela classe, ou seja, desconsidera
a informação provida pela correlação entre atributos. Nas demais variantes o problema foi contornado devido à regularização das matrizes de covariância. Sendo elas:

- Regularização Tikhonov

![alt text](image.png)

onde 0 < lambda << 1
- Regularização usando a matriz de convariancia agregada (Pooled)

![alt text](image-2.png)

- Regularização de Friedman

![alt text](image-1.png)