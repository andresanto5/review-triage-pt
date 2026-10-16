# Interpretação do primeiro experimento

## Pergunta

É possível reconhecer textos associados a notas 1 ou 2 usando somente título e corpo da avaliação, em avaliações posteriores ao período de treino?

## Resultado observado

Teste de 8 a 31 de maio de 2018, com 18.546 avaliações:

| Medida | TF-IDF + regressão logística |
|---|---:|
| Macro-F1 | 0,9247 |
| Precisão da classe negativa | 0,8705 |
| Recall da classe negativa | 0,9180 |
| Average precision | 0,9417 |
| Encaminhadas à revisão | 30,15% |

Referência que sempre prevê a classe majoritária: macro-F1 0,4166 e recall zero para notas 1/2.

Dos 5.303 registros negativos, 4.868 foram identificados e 435 ficaram fora da fila. Outros 724 registros de notas 3/4/5 entraram na fila. São erros contra estrelas, não contra julgamento de especialistas.

O limiar de 0,625 maximiza macro-F1 na validação. Não é uma decisão validada de custo/capacidade: antes de uso, definir custo dos erros e tamanho da fila e escolher o limiar na validação com esse objetivo. Não otimizar novamente no teste já observado.

## Auditoria

132.373 linhas originais; 546 inválidas ou curtas removidas; 293 linhas em 35 grupos de texto com alvos conflitantes removidas; 2.436 duplicatas normalizadas removidas. Restaram 129.098 linhas.

Deduplicação global e exclusão de conflitos usam o conjunto completo para curadoria antes da divisão. Isso evita textos idênticos entre partições, mas seleciona uma amostra mais limpa e não replica fielmente uma ingestão produtiva. Avaliar posteriormente o fluxo cronológico bruto como cenário separado, sem reaproveitar este teste para decisões de modelo.

O resultado sustenta um baseline útil nesta base histórica. Não mede satisfação atual, ganho financeiro, equidade ou qualidade em produção. Termos avaliativos no título podem ser fortemente associados à nota; comparar título+corpo versus corpo apenas na validação é uma próxima análise relevante.

## Continuação concluída

O registro retrospectivo no MLflow, os diagnósticos agregados, a comparação título/corpo na validação e a API local com testes de contrato foram concluídos. Consulte [ANALISE.md](ANALISE.md) e o README. Qualquer segundo modelo exige protocolo de comparação antes de nova medição final.

Nenhum resultado de comparação com embeddings, calibração, monitoramento ou operação em produção foi produzido nesta etapa.
