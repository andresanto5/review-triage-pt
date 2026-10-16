# Validação da API local

Execução em 16/10/2026, Docker local, sem AWS ou APIs pagas.

- Oito testes automatizados passaram, incluindo os dois testes de dados existentes.
- Testes HTTP conferem paridade com o modelo congelado, metadados, JSON inválido, limites de tamanho, campos extras, tipos inválidos e respostas sem eco de conteúdo.
- Inicialização rejeita modelo ausente e limiar inválido.
- Docker Compose iniciou a API com modelo e código montados somente para leitura.
- Verificação pelo Windows confirmou `/health` pronto, `/predict` respondendo e `/docs` com HTTP 200.
- Um exemplo sintético de produto quebrado foi encaminhado para revisão humana. Isso demonstra o contrato funcionando; não acrescenta evidência de desempenho preditivo.

SHA-256 do modelo servido: `2f1987cdc215e76800d5c54310c9a4621fc581b5b839eb8b4b4abb18a6bdb9a5`.

O baseline e seu limiar não foram alterados. Não foram feitos teste de carga, autenticação, publicação externa ou validação em produção. A interface está em `http://127.0.0.1:8000/docs` enquanto o contêiner estiver ativo; use `docker compose down` para encerrar.
