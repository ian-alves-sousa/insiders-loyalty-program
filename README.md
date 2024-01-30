# Customer Loyalty Program for an E-Commerce

<div align="center">
<img src="img/capa.png" />
</div>

# Introdução

Este é um projeto end-to-end de Data Science de um modelo de clusterização. No qual criamos um script focado em encontrar os melhores clientes que irão participar do grupo Insiders. A clusterização foi colocado em produção na AWS.

Este repositório contém a solução para a resolução de um problema do Kaggle, contudo, o mesmo não está mais disponível para resolução, e portanto, sua página foi desativada.

Esse projeto faz parte da "Comunidade DS", que é um ambiente de estudo que promove o aprendizado, execução, e discussão de projetos de Data Science.

### Plano de Desenvolvimento do Projeto de Data Science

Esse projeto foi desenvolvido seguindo o método CRISP-DS(Cross-Industry Standard Process - Data Science). Essa é uma metodologia capaz de transformar os dados da empresa em conhecimento e informações que auxiliam na tomada de decisão. A metodologia CRISP-DM define o ciclo de vida do projeto, dividindo-as nas seguintes etapas:

- Entendimento do Problema de Negócio
- Coleção dos Dados
- Limpeza de Dados
- Análise Exploratória dos Dados
- Preparação dos Dados
- Modelos de Machine Learning, Cross-Validation e Fine-Tuning.
- Avaliação dos Resultados do Modelo e Tradução para Negócio.
- Modelo em Produção

![crisp!](img/crisp.png)

### Planejamento

- [1. Descrição e Problema de Negócio](#1-descrição-e-problema-de-negócio)
- [2. Base de Dados e Premissas de Negócio](#2-base-de-dados-e-premissas-de-negócio)
- [3. Estratégia de Solução](#3-estratégia-de-solução)
- [4. Top Insights](#4-top-insights)
- [5. Seleção do Modelo de Machine Learning](#5-seleção-do-modelo-de-machine-learning)
- [6. Resultados de Negócio](#6-resultados-de-negócio)
- [7. Modelo em Produção](#7-modelo-em-produção)
- [8. Conclusão](#8-conclusão)
- [9. Aprendizados](#9-aprendizados)
- [10. Trabalhos Futuros](#10-trabalhos-futuros)

# 1. Descrição e Problema de Negócio

### 1.1 Descrição

A empresa All in One Place é uma empresa Outlet Multimarcas, comercializa produtos de segunda linha de várias marcas a um preço menor, através de um e-commerce.

### 1.2 Problema de Negócio

A empresa All in One Place é uma empresa Outlet Multimarcas, comercializa produtos de segunda linha de várias marcas a um preço menor, através de um e-commerce.

Em um pouco mais de 1 ano de operação, o time de marketing percebeu que alguns clientes da sua base compram produtos mais caros, com alta frequência e acabam contribuindo com uma parcela significativa do faturamento da empresa.

Baseado nessa percepção o time de marketing vai lançar um programa de fidelidade para os melhores clientes da base, chamado **Insiders**, mas o time não tem um conhecimento avançado em análise de dados para eleger os participantes do programa.

**Para tal desenvolverei um produto de dados que determine os clientes elegíveis permitindo ao time de Marketing tomar ações personalizadas e exclusivas ao grupo, de modo a aumentar o faturamento e frequência de compra.**

### 1.3 Expectativas

Os gestores da All in One Place esperam poder:

- Saber quem são os clientes Insiders;
- Saber aqal o comportamento de compra do cliente Insider;
- Monitorar a mudança dos clusters;
- Visualizar os valores dos cluters e suas mudanças em tempo real;
- Decidir quais serão as estratégias usadas no programa de fidelidade.

# 2. Base de Dados e Premissas de Negócio

## 2.1 Base de Dados

O conjunto de dados total possui as seguintes informações:
Onde cada linha é referente a uma transação (produto comprado).

| **Feature**            | **Description**                                                                   |
| ---------------------- | --------------------------------------------------------------------------------- |
| invoice_no             | id da compra                                                                      |
| stock_code             | id do produto                                                                     |
| description            | descrição do produto                                                              |
| quantity               | quantidade comprada do produto                                                    |
| invoice_date           | data da compra                                                                    |
| unit_price             | preço unitário do produto                                                         |
| customer_id            | id do cliente                                                                     |
| country                | país de entrega                                                                   |

## 2.2 Premissas de Negócio

Com base em pesquisa de mercado foram tomadas as seguintes suposições de negócio:

- Remoção de itens com preço inferior a 0.04.
- Tirar da base os cliente não identificados e aqueles que tem apenas uma compra no período específicado.
- Itens com quantidade negativa ou com o número do pedido destacando a letra 'C' serão considerados estornos.
- Códigos de estoque como 'POST', 'D', 'DOT', 'M', 'S', 'AMAZONFEE', 'm', 'DCGSSBOY','DCGSSGIRL', 'PADS', 'B', 'CRUK' serão removidos por não haver clareza quanto a sua finalidade.
- Removidas compras de clientes para regiões não especificadas ou inconclusivas.

# 3. Estratégia de Solução

A estratégia de solução foi a seguinte:

### Passo 01. Descrição dos Dados

Coleto os dados e realizo uma breve análise e suas estatísticas, também limpo alguns dados com potenciais comprometedoras, o meu objetivo se concentra está em ganhar conhecimento inicial do problema em que estou lidando e começar a planejar quais ferramentas devo utilizar na manipulação para os algoritmos.

### Passo 02. Filtragem de Variáveis

Nesta etapa busco remover algumas variáveis criadas para auxiliar o processo de Feature Engineering, também removo a coluna 'description' por não conter informação relevante.

### Passo 03. Featuring Engineering

Desenvolvo hipóteses iniciais sobre o negócio para poder derivar novos atributos com base nas variáveis originais para descrever melhor o fenômeno a ser compreendido, estes atributos podem me auxiliar na validação de hipóteses e no treinamento do modelo de Machine Learning.

### Passo 04. Análise Exploratória dos Dados (EDA)

Realizo estudos das variáveis univariadas e como os dados se comportam bivariados, também busco compreender o comportamento de cada variável e suas correlações com as demais.

### Passo 05. Preparação dos Dados

Sessão que trata da preparação dos dados para que os algoritmos de Machine Learning possam ser aplicados. Foram realizados alguns tipos de escala e encoding para que as variáveis categóricas se tornassem numéricas.

### Passo 06. Estudo do Espaço

Realizo transformações do espaço de dados com o objetivo de gerar uma melhor segmentação dos clientes afim de encontrar os melhores perfis.

### Passo 06. Seleção de Variáveis do Algoritmo

Na seleção dos atributos foi realizado um estudo de importância das colunas, no qual os atributos mais significativos foram selecionados para um estudo mais aprofundado com intuito de gerar melhor entendimento e criar features que separam melhor os dados.

### Passo 07. Modelo de Machine Learning

Realização do treinamento dos modelos de Machine Learning. O modelo que apresentou melhor Silhouette Score com um número de cluster igual a 8 foi escolhido.

### Passo 08. Hyper Parameter Fine Tuning

Foi encontrado os melhores parâmetros que faziam a clusterização convergir para o mesmo valor sempre.

### Passo 09. Cluster Analysis

Através dos clusters gerados pelo modelo, uma análise foi feita com o intuito de entender melhor as características de cada cluster, identificar o cluster Insiders e responder as perguntas de negócio.

### Passo 9. Deploy do Modelo em Produção

Nesta etapa o projeto é disponibilizado via Dashboard no Metabase com as informações de cada grupo para o programa de fidelidade. Com isso, é desenvolvido uma arquitetura na AWS para colocar esse projeto em produção, com o banco, dataset, encoders, modelos e máquina virtual direto na cloud.

# 4. Top Insights

----------------- PAREI AQUI --------------------------

## 4.1 Análise Univariada

- Variáveis Numéricas: o histograma abaixo mostra como está organizada a distribuição das variáveis numéricas do nosso conjunto de dados. Mostra a contagem de cada variável numérica do dataset.

<p align="center">
  <img src="img/uni.png" alt="Numerical-Variables">
</p>

## 4.2 Análise Bivariada

Por ser um Hackday, nessa etapa gerados hipoteses para validar, entender melhor o negócio e criar features ou filtrar os dados para melhorar o RMSE.

### H1 - Semanas com mais desconto tem maiores vendas semanais

**VERDADEIRO** As semanas que apresentam descontos, apresentam também maiores vendas semanais.

- No primeiro gráfico podemos ver que a frequência das vendas tem um pico no início, mas mostra um grupo de outliers com valores mais altos.
- O mesmo acontece no segundo gráfico, onde vemos um grupo que se distancia bastante do terceiro quartil.
- Isso pode evidenciar exatamento as semanas que apresentam desconto, e com isso, tem um comportamento mais atípico quando comparado com os demais meses do ano, mostrando assim a importância dessa coluna para a previsão das vendas semanais das últimas semanas do ano.


![H1!](img/H1.png)

### H5 - Junho é o mês com menor vendas semanais.

**FALSO** - O mês com a menor média de vendas semanais é Janeiro e com a melhor é Julho. (De Janeiro a Outubro)

- No gráfico podemos observar em vermelho as semanas do mês de Janeiro, evidenciando o mês com a menor média nas vendas semanais e o melhor mês em Julho.
- Dessa forma, é justo imaginar que os clientes não compram em janeiro por estarem guardando o dinheiro do começo do ano para constas como IPTU e IPVA, além disso, eles podem ter usado os bônus de fim de ano para comprar exatamente onde queremos prever, a Balck Friday e o Natal, mostrando a importância do estudo.

![H2!](img/H2.png)

### H3 - Eletrodoméstico vende mais que Eletrônico.

**VERDADEIRO** - Eletrodomésticos apresentam vendas semanais maiores que eletrônicos.

- No gráfico observamos que os clientes tem preferência por comprar eletromésticos, sendo esse o tipo de produto que é a cara da EletroPlaza quando os clientes pensam na marca.

![H3!](img/H3.png)

## 4.3 Análise Multivariada

Essa etapa mostra como cada coluna do dataset está relacionada entre si e nos gera ideias para criar features e testar.

<p align="center">
  <img src="img/MAPA.png" alt="MAPA">
</p>

Apesar da pouca correlação, através desse mapa de calor conseguimos destacar que algumas features apresentam as maiores relações com a nossa features preditiva:
- Tamanho da loja;
- Setor;
- Loja/tipo.

### 4.3.1 Features Importance

Além disso, foi gerado um Feature Importance para entender quais features um modelo de árvore mais usaria para prever a variável resposta.

<p align="center">
  <img src="img/FI.png" alt="FI">
</p>

Percebemos assim que a **magnitude das vendas semanais está extremamante relacionada ao seu setor**, e o tamanho e tipo ta loja também influenciam na previsão.

# 5. Seleção do Modelo de Machine Learning

Os seguintes algoritmos de Machine Learning foram aplicados:

- Linear Regressor;
- Linear Regressor - Lasso;
- Linear Regressor - Ridge;
- Linear Regressor - Elastic Net;
- K-NearestNeighbors Regressor;
- Random Forest Regressor;
- Light Gradient Boosting Machine Regressor;
- XGBoost Forest Regressor;
- Gradient Boosting Regressor;

O método de cross-validation foi utilizado em todos os modelos.

# 6. Resultados de Negócio

Para medir o desempenho dos modelos, usaremos o método de validação cruzada que evita que o modelo seja superajustado quando o modelo recebe alguns dados que nunca viu antes (garantindo a generalização). 

A real performance dos modelos utilizando método CROSS-VALIDATION.

<p align="center">
  <img src="img/comparacao.png" alt="Comparação">
</p>

Como a competição nos instigava para encontrar o menor RMSE, essa foi a métrica que focamos melhorar. Assim, os modelos que apresentam o menor RMSE no Cross Validation foram:
- Random Forest Regressor;
- Light Gradient Boosting Machine Regressor;

Dessa forma, ambos os modelos passaram por um **HYPERPARAMETER FINE TUNING**, onde foram testados diversos pâmetros para esses modelos e os que apresentavam a melhor métrica após o  Cross Validation era usado para gerar o arquivo de submissão, segue abaixo um exemplo do quão perto os valores gerados foram.

<p align="center">
  <img src="img/RS.png" alt="RS">
</p>

# 7. Modelo em Produção

Esse processo foi repetido durante os dois dias da competição. Ao criar uma nova feature ou gerar um novo Insight depois da Exploração dos Dados todo o modelo era treinado novamente e a métrica avaliada, caso o erro fosse menor que o da última submissão, uma nova tentativa era feita.

## 7.1 Tentativas

Assim, algumas ações foram feitas e testadas, segue abaixo o resultado prático:

| **Ação** | **Resultado** |
| ------------------------------------------- | ------------- |
| Excluir os registros com vendas semanais nula ou menor que 1 | Melhorou |
| Retirar os descontos        | Piorou |
| Preencher os registros com distância faltantes usando um número grande (sem concorrencia)| Estável|
| Criar colunas de dias e mês| Melhorou|
| Classificar as lojas pelo tamanho (grande, médio e pequeno)| Melhorou|
| Somar os descontos em uma coluna| Melhorou|

Assim, a ação que nos fez dar um salto na pontuação foi **Excluir os registros com vendas semanais nula ou menor que 1**.

## 7.2 Insights Acionáveis

Através do projeto e da análise alguns Insights Acionáveis foram criados:


**Janeiro apresenta a pior média de vendas semanal e julho a melhor.**
 - Com essa informação, podemos planejar promoções específicas ou lançamentos de produtos no mês de julho, visto que ele apresenta uma boa performance. Sobre janeiro, podemos identificar as campanhas e gerar estratégias de marketing como um saldão pós-ano novo para atrair os clientes e melhorar as vendas desse mês.

**Eletrodomésticos têm vendas superiores a eletrônicos e outros tipos de produtos.**
- Concentrar esforços de marketing e estoque em eletrodomésticos, talvez com promoções especiais ou pacotes que incluam esses produtos. Com isso, podemos ter um estoque adequado desses tipos de produto, visto que é o carro chefe da empresa e lidera nas vendas.

## 7.3 Ranking Final

A partir dessa construção e de todas as tentativas ficamos na terceira posição como o terceiro menor RMSE tanto no Leaderbord Público quanto Privado (que foi liberado apenas no final da competição):

<p align="center">
  <img src="img/FINAL.png" alt="FINAL">
</p>

E com esse resultado nos classificamos para a final, que ocorreu 3 dias depois, onde apresentamos o slide do link a seguir:

<a href="https://github.com/ian-alves-sousa/hackday_7_sales_prediction_regression/blob/main/Apresentação_%20Eletro%20Plaza%20Store.pdf" target="_blank">Apresentação Final</a>

O contexto do Grinch é que as duas outras equipes que se classificaram para a final tinham nome relacionado ao Natal, dessa forma, fomos para acabar com a comemoração.

# 8. Conclusão

Neste projeto, todas as etapas necessárias foram realizadas para implementar um projeto completo de Ciência de Dados em uma competição de dados. Foi utilizado o método de gerenciamento de projetos denominado CRISP-DM/DS e obteve-se a terceira menor métrica (RMSE) que levou a equipe para a final do Hackday 7 da Comunidade DS.

Tendo em vista esses resultados, o projeto alcançou seu objetivo de encontrar uma solução simples e assertiva para previsão de vendas semanais, realizando o projeto em apenas dois dias. E o principal foi melhorar 283% nesse meio tempo.

# 9. Aprendizados

**Aprendizados**

- Esse problema de regressão no Hackday 7 teve um formato diferente, onde os dados estavam com muitos valores faltantes, e uma análise de dados bem feita seria crucial para conseguir os resultados. Isso nos mostrou a importância de entender o contexto do negócio e os dados para a construção do projeto.

- Assim, a Análise Exploratória de Dados se demonstrou uma das etapas mais importantes do projeto, pois é nessa parte que podemos encontrar Insights de Negócio que promovem novos conhecimentos e até contradições que nos fazem repensar o negócio como um tudo. E através dela que tivemos as ideias para filtragem e criação de features.

- Com pouco tempo para testes, uma boa organização supera a falta de conhecimento. Foi com uma boa divisão de tarefas que o time conseguiu concluir todas as etapas do projeto.

**Trabalhos Futuros**

- Criar novas features, afim de explicar com mais eficiência o os fenômenos do problema e consequentemente gerar resultados melhores.
- Testar diferentes Encoders na preperação dos dados.
- Aplicar o balanceamento dos dados e entender como isso influencia na resolução desse problema.
- Traduzir os valores dos erros em US$, contendo o melhor e o pior cenário por loja, para melhor visualização e análise do time de negócio;
- Aprofundar a compreensão do desempenho das lojas com base em seus tipos específicos de produtos, especialmente as lojas de eletrodomésticos. O objetivo é extrair insights que nos permitam otimizar a alocação de recursos, concentrando nossos esforços em áreas estratégicas que impulsionem o desempenho global da EletroPlaza Store.

# 10. Trabalhos Futuros

<a href="https://www.linkedin.com/in/christianods/" target="_blank">Christiano Bruneli Peres</a><br>
<a href="https://www.linkedin.com/in/ian-alves-sousa/" target="_blank">Ian Alves Sousa</a><br>
<a href="https://www.linkedin.com/in/paulawehdorn/" target="_blank">Paula Wehdorn Wildemberg</a><br>
<a href="https://www.linkedin.com/in/victor-bongestab/" target="_blank">Victor Bongestab</a><br>
<a href="https://www.linkedin.com/in/vinicius-gasperazzo-rosa/" target="_blank">Vinicius Gasperazzo Rosa</a>
