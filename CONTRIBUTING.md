# Contribuir com o A.C.E.

## Preparar o ambiente

Clone este repositório pelo botão **Code** do GitHub e abra a pasta do clone no PowerShell. Você precisa de Git, uv e Python 3.10 a 3.13. Para a interface, instale Node.js 22.22 ou mais recente e npm 11.19 ou mais recente da versão 11. Para compilar o aplicativo nativo no Windows, instale Rust, o toolchain MSVC e as dependências do Tauri. [Instalar uv](https://docs.astral.sh/uv/getting-started/installation/) · [Instalar dependências do Tauri](https://v2.tauri.app/start/prerequisites/)

Instale as dependências e inicie o servidor em um perfil descartável. A porta 8001 deixa livre a porta padrão 8000 da instalação normal.

```powershell
uv sync --python 3.13 --extra dev --extra server
$env:OPENJARVIS_HOME = Join-Path $env:TEMP ("ace-dev-" + [guid]::NewGuid().ToString())
Remove-Item Env:OPENJARVIS_CONFIG -ErrorAction SilentlyContinue
uv run jarvis init --preset ace --no-download --no-analytics
uv run jarvis serve --host 127.0.0.1 --port 8001 --local-services-only
```

Em outro PowerShell, na raiz do repositório, inicie a interface e abra o endereço indicado pelo Vite.

```powershell
Set-Location frontend
npm ci
$env:OPENJARVIS_VITE_PROXY_TARGET = 'http://127.0.0.1:8001'
npm run dev
```

Esse ambiente permite trabalhar nas telas e APIs locais sem credenciais ou modelos. Use o aplicativo empacotado para testar seleção de provedor, bandeja do Windows e reaproveitamento do servidor existente. [Guia de arquitetura e uso do A.C.E.](docs/user-guide/ace.md)

## Onde alterar

| Caminho | Responsabilidade |
| --- | --- |
| `frontend/src/` | Interface React, configuração de provedores e visualização da memória. |
| `frontend/src-tauri/` | Janela nativa, bandeja, inicialização e verificação do backend. |
| `src/openjarvis/engine/` e `src/openjarvis/server/` | Provedores, descoberta de modelos e API. |
| `src/openjarvis/memory/`, `src/openjarvis/proactive/` e `src/openjarvis/tools/` | Persistência, monitoramento e ferramentas. |
| `tests/` e `frontend/src/` | Testes de backend e da interface. |
| `deploy/windows/` e `scripts/install/build_ace_backend.py` | Inicializador e preparação do pacote para Windows. |

## Verificar uma alteração

Rode os testes da área que mudou. Este conjunto cobre a inicialização local e a integração com a Cloudflare usando respostas controladas.

```powershell
uv run pytest tests/server/test_local_services.py tests/server/test_cloudflare_chat.py tests/engine/test_cloudflare.py -q
uv run ruff check src tests
Set-Location frontend
npm test
npm run build:tauri
```

Se alterar o contrato entre a interface nativa e o backend, confira a revisão da API em `frontend/src-tauri/src/lib.rs` e `src/openjarvis/server/daemon.py`. Para validar a distribuição, teste uma instalação limpa, a atualização sobre a versão anterior e a abertura sem Cloudflare. Teste pelo menos um provedor local com modelo instalado. Registre no pull request o sistema, o modo usado e quais testes chegaram a rodar. [Empacotamento para Windows](docs/getting-started/installation.md#ace-for-windows-x64)

## Enviar uma contribuição

1. Abra uma branch com uma mudança que possa ser revisada separadamente.
2. Mantenha o modo local utilizável sem rede ou credenciais. Solicite autorização antes de ações que alterem arquivos, aplicativos ou o sistema.
3. Use dados fictícios em testes e relatos. Mantenha tokens, arquivos `.env`, perfis locais, bancos de memória, logs pessoais, modelos e instaladores fora do Git.
4. Abra o pull request com o problema, como reproduzi-lo, a solução e os testes. Se mudar a instalação ou uma opção de uso, atualize também o guia correspondente.

Para alterações em componentes compartilhados com o OpenJarvis, consulte as [convenções do projeto de origem](docs/development/contributing.md). Preserve a atribuição e os avisos de licença nos arquivos derivados.
