# Diagnóstico e comparação na validação

## Onde o baseline erra

Diagnóstico retrospectivo do teste original, sem ajustar o modelo ou seu limiar:

| Nota | Avaliações | Erros | Taxa de erro |
|---|---:|---:|---:|
| 1 | 4.013 | 156 | 3,89% |
| 2 | 1.290 | 279 | 21,63% |
| 3 | 2.349 | 498 | 21,20% |
| 4 | 4.450 | 112 | 2,52% |
| 5 | 6.444 | 114 | 1,77% |

A concentração nas notas 2/3 é compatível com maior dificuldade perto da fronteira do alvo, mas não comprova ambiguidade sem inspeção e anotação humana. As notas continuam sendo um proxy. Notas 2 geram falsos negativos; notas 3 geram falsos positivos pela definição adotada.

Textos normalizados com até 50 caracteres tiveram 18,08% de erro (260 casos); os de mais de 400, 13,86% (772 casos). Os intervalos intermediários tiveram 4,22% e 9,19%. Não há demonstração causal de efeito do comprimento; composição das classes e amostras diferentes podem explicar parte disso.

As quatro semanas observadas tiveram taxas entre 5,04% e 6,91%, com primeira/última semanas parciais. Isso não constitui monitoramento nem teste de estabilidade temporal. CSVs agregados acompanham este relatório; nenhum texto de consumidor é exportado.

## Quanto o título contribui

Treino comum: 89.111 registros. Validação comum: 19.403 registros. Corpos idênticos não se repetem entre treino e validação. Ambos os modelos mantêm TF-IDF + regressão logística e escolhem limiar exclusivamente na validação.

| Atributos | Macro-F1 | Precisão de notas 1/2 | Recall de notas 1/2 |
|---|---:|---:|---:|
| Título + corpo | 0,9282 | 0,8712 | 0,9188 |
| Somente corpo | 0,9058 | 0,8394 | 0,8848 |

O título contribui nesta amostra: diferença descritiva de 0,0223 em macro-F1. Não foi realizado teste de significância. Os resultados incluem seleção de limiar e avaliação na mesma validação e podem ser otimistas. Nenhuma das variantes novas foi avaliada no teste. Esta amostra difere da avaliação original e seus números não devem ser comparados diretamente com o macro-F1 original do teste.

A curadoria global do baseline permanece: exclusão de conflitos usa rótulos de todo o período antes da divisão. Portanto, o protocolo mede uma amostra curada, não uma ingestão produtiva fiel. Um futuro protocolo cronológico bruto deve ser definido antes de nova avaliação final.

## Rastreabilidade e próximo passo

MLflow local registra três execuções: importação retrospectiva do baseline e duas variantes na validação. Modelos, relatórios agregados, parâmetros e hash da base foram salvos. O código de treinamento disponível agora tem hash registrado, sem alegação de recuperação da versão exata executada originalmente.

A API local do baseline congelado foi implementada com validação de entrada e testes HTTP de contrato; instruções no README. A comparação com embeddings pode vir depois, sob protocolo definido e sem retunar no teste observado. Não há evidência de ganho financeiro, calibração ou desempenho em produção.
