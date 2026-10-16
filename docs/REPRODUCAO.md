# Reprodução completa — Review Triage PT

Execute todos os comandos abaixo na raiz do projeto, não na pasta `docs/`. Para a apresentação resumida, consulte o [README](../README.md).

Projeto de portfólio: priorizar avaliações de produtos com notas 1 ou 2 para revisão humana. A etiqueta deriva das estrelas e é um proxy, não anotação humana de sentimento ou comprovação de reclamação procedente.

## Primeira etapa

Baseline TF-IDF + regressão logística, separação temporal em treino/validação/teste, remoção de textos duplicados antes da separação e escolha de limiar apenas na validação. Comparação com predição da classe majoritária. Sem AWS, API paga ou dados da empresa.

Usa somente título e texto como atributos. Nota define o alvo e não entra nas features. Identificadores, sexo, idade e localização não são usados. Não publica textos de usuários nem erros com conteúdo bruto.

## Dados e atribuição

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), disponibilizado por B2W Digital; autores Livy Real, Marcio Oshiro e Alexandre Mafra. Avaliações históricas de 2018. Licença da base: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), com atribuição, uso não comercial e compartilhamento de derivados sob a mesma licença. A base e os modelos derivados não são incluídos no repositório. Não confundir dados públicos com ausência de restrições de uso.

Download original: https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/main/B2W-Reviews01.csv

Salvar em `data/raw/B2W-Reviews01.csv`. O cabeçalho do arquivo baixado usa vírgulas, apesar da indicação de ponto e vírgula no README original. O relatório registra SHA-256 do arquivo para reprodução.

## Executar

Python 3.12, dependências em requirements.txt:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_data.py
python src/train.py
python -m unittest discover -s tests
```

Resultados agregados em `reports/`; modelo e auditoria privada em `artifacts/` (ignorados pelo Git).

## Limites e próximas etapas

- Dados antigos, sem validação em distribuição atual.
- Deduplicação exata normalizada não elimina paráfrases, todos os templates ou sobreposição de produtos.
- Sem ajuste de hiperparâmetros no teste; qualquer nova decisão deve usar treino/validação ou outro teste final independente.
- Antes de operação: avaliação de calibração, capacidade da fila e critérios de encaminhamento humano.
- Implementados: MLflow local, análise agregada de erros, comparação título/corpo na validação e ambiente Docker.
- API local implementada com validação de entrada, identificação do modelo e testes HTTP de contrato.
- Próxima comparação possível: embeddings, ainda não implementada.

## Experimentos locais

```powershell
docker build -t review-triage-pt:local .
docker run --rm --network none --mount "type=bind,source=$($PWD.Path),target=/project" -w /project review-triage-pt:local python src/train.py
docker run --rm --network none --mount "type=bind,source=$($PWD.Path),target=/project" -w /project review-triage-pt:local python src/analyze.py
```

`analyze.py` importa retrospectivamente o baseline e registra duas variantes novas, avaliadas somente na validação. As duas usam as mesmas 89.111 linhas de treino e 19.403 de validação, com corpos idênticos removidos entre partições. Os limiares são escolhidos nessa validação: os números servem à seleção de modelo, não a uma estimativa final independente.

| Variante | Macro-F1 na validação | Recall de notas 1/2 |
|---|---:|---:|
| Título + corpo | 0,9282 | 0,9188 |
| Somente corpo | 0,9058 | 0,8848 |

Metadados SQLite e modelos ficam em `artifacts/`, ignorados pelo Git. Três execuções concluídas, com métricas e artefatos verificados por `python src/verify_tracking.py`; identificadores em `reports/mlflow_runs.json`. Versões instaladas no ambiente Linux/Python 3.12 constam em `requirements.lock.txt`. O hash `train_script_sha256` registra o arquivo disponível na análise retrospectiva, não comprova a versão exata do treinamento original. O hash da base é conferido contra o modelo salvo e o relatório original.

Para abrir a interface local, mantendo o mesmo caminho de montagem:

```powershell
docker run --rm -p 127.0.0.1:5000:5000 --mount "type=bind,source=$($PWD.Path),target=/project" -w /project review-triage-pt:local mlflow ui --backend-store-uri sqlite:////project/artifacts/mlflow.db --host 0.0.0.0 --port 5000
```

Acesse http://127.0.0.1:5000. Experimentos usam CPU local; não exigem AWS ou APIs pagas. Consulte [a análise](../reports/ANALISE.md) para as limitações dos resultados.

## API local

Pré-requisito: executar o treinamento para gerar `artifacts/baseline.joblib`. O Compose monta esse modelo e o código como somente leitura, sem montar a base de dados ou os registros do MLflow.

```powershell
docker compose up --build -d
Invoke-RestMethod http://127.0.0.1:8000/health
$review = @{ title = 'Produto com defeito'; text = 'O produto chegou quebrado e não funciona.' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($review))
docker compose down
```

Interface interativa em http://127.0.0.1:8000/docs. A porta fica disponível somente no computador local. A API recebe `title` opcional e `text` obrigatório; rejeita campos extras, tipos incorretos, corpo vazio, título acima de 500 caracteres, texto acima de 10.000 e entrada normalizada com menos de 10 caracteres. O limite HTTP é 64 KiB; JSON inválido retorna 422, conteúdo sem `application/json` retorna 415 e requisição grande retorna 413.

`negative_score` é um escore **não calibrado** do proxy de notas 1/2. `review_required` aplica o limiar congelado de 0,625 para encaminhamento humano. A resposta inclui o SHA-256 do arquivo do modelo e não inclui o texto da avaliação. A API usa a mesma normalização do treinamento, não modifica o modelo e não registra corpos de requisição. O servidor inicia apenas se conseguir carregar um modelo válido. Carregue somente artefatos joblib confiáveis, pois esse formato pode executar código durante a leitura.

Testes (exigem o baseline local):

```powershell
docker run --rm --network none --mount "type=bind,source=$($PWD.Path),target=/project" -w /project review-triage-pt:local python -m unittest discover -s tests -v
```

Cobertura: paridade com a inferência original, impressão digital do modelo, contrato de entrada/saída, erros sem eco de conteúdo, tamanho da requisição e falha de inicialização sem modelo válido. Não há autenticação, validação de capacidade ou implantação pública; esta é uma demonstração local de inferência.
