# Ficha do modelo

| Item | Descrição |
|---|---|
| Uso pretendido | Demonstração de priorização de avaliações em português para revisão humana |
| Alvo | 1 para notas 1/2; 0 para notas 3/4/5 |
| Entradas | Título opcional e texto da avaliação |
| Modelo | TF-IDF unigramas/bigramas, até 50 mil atributos; regressão logística balanceada |
| Treino | 01/01 a 16/04/2018; 90.870 registros após curadoria |
| Validação | 17/04 a 07/05/2018; 19.682 registros |
| Teste | 08/05 a 31/05/2018; 18.546 registros |
| Limiar | 0,625, selecionado por macro-F1 na validação |
| Saída | Escore não calibrado e encaminhamento booleano |
| Identificação | SHA-256 do arquivo servido em `/health` e `/predict` |

## Avaliação

Macro-F1 no teste: 0,9247. Precisão de notas 1/2: 0,8705; recall: 0,9180. [Números completos e auditoria](../reports/baseline.json).

A comparação título/corpo foi feita somente na validação, numa amostra comum diferente. Nenhum novo modelo foi avaliado no teste observado.

## Restrições

As estrelas não equivalem a sentimento anotado por especialistas. Nota 3 entra na classe não negativa por definição. Dados de 2018 não comprovam qualidade atual. A exclusão global de conflitos usa rótulos de todo o período antes da divisão, selecionando amostra mais limpa que a ingestão produtiva. Não se eliminaram todas as paráfrases ou sobreposições de produtos.

Não se mediram calibração, equidade por grupos, custos de erro, impacto financeiro, latência sob carga ou desempenho em produção. Ausência de atributos demográficos nas entradas não comprova ausência de vieses no texto.

## Uso e dados

Não usar como decisão autônoma sobre consumidores nem como evidência de reclamação procedente. API demonstrativa local, sem autenticação. Carregar somente arquivos joblib confiáveis.

Dados: B2W-Reviews01, B2W Digital, Livy Real, Marcio Oshiro e Alexandre Mafra, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Modelos derivados e base ficam locais; observe as restrições da fonte. O repositório contém código e relatórios agregados, sem textos de consumidores.
