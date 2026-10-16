# Demonstração em três minutos

## Problema e decisão — 30 segundos

“Quero priorizar avaliações para revisão humana. Uso notas 1/2 como proxy e testo primeiro um modelo simples de texto. A saída é uma sugestão de fila, sem automatizar a decisão de atendimento.”

## Evidência e limites — 60 segundos

Abra `reports/INTERPRETACAO.md`. Mostre macro-F1 0,9247 e a referência majoritária de 0,4166. Explique 435 casos negativos perdidos e 724 outros encaminhados. O corte é temporal e o limiar vem da validação. A curadoria global usa rótulos antes da divisão e os dados são de 2018; isso limita a interpretação.

Abra `reports/ANALISE.md`: erros nas notas 2/3 e comparação título/corpo na validação. Não apresente essa comparação como uma nova medição de teste.

## Uso do modelo — 60 segundos

Com o treinamento executado, rode `docker compose up -d` e abra `http://127.0.0.1:8000/docs`.

Em `POST /predict`, clique em **Try it out**, use o exemplo preenchido e clique em **Execute**. Explique `review_required`, o limiar e o hash do modelo. O escore não é calibrado. Envie depois `{"text":"curto"}` para demonstrar a validação 422.

Alternativa: `docker compose exec -T api python src/demo.py`. O exemplo é sintético, sem texto de consumidor e sem constituir nova evidência de desempenho.

## Engenharia e evolução — 30 segundos

Mostre os testes e os registros do MLflow, se tiver executado `analyze.py`. A API usa um modelo congelado e os testes conferem paridade. A execução é local, sem AWS. Próximas perguntas: capacidade da fila, calibração e avaliação em novo período sob protocolo definido antes do resultado final.

Não afirmar experiência de produção AWS, redução de custos, impacto no atendimento ou implantação produtiva com base nesta demonstração.
