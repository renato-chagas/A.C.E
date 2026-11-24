# A.C.E.

O A.C.E. é um assistente para Windows com conversas, memória e pesquisa. Você pode usar um modelo no seu computador ou ativar a Cloudflare para conversar online. A memória, o grafo de conhecimento e as fontes que você salvou ficam no computador e podem ser consultados sem modelo.

## [Baixar o instalador do A.C.E. para Windows (.exe)](https://github.com/renato-chagas/A.C.E/releases/latest/download/Instalar-A.C.E.exe)

Abra `Instalar-A.C.E.exe` no Windows de 64 bits e clique em **Instalar**. O arquivo inclui o aplicativo, o serviço local em Python e o WebView2. Mantenha a opção de abrir o A.C.E. marcada ao concluir.

Na primeira abertura, escolha **Configure model** para ativar um modelo de conversa ou **Continue with local services** para usar a memória e consultar as fontes salvas.

Para atualizar, saia do A.C.E. pelo ícone perto do relógio, execute o novo instalador e mantenha a pasta da instalação existente. [Guia de instalação e atualização](docs/getting-started/ace-windows.md)

## Escolher como conversar

| Opção | Quando usar |
| --- | --- |
| **Local Ollama** | Para usar um modelo já baixado no Ollama. [Configurar Ollama](docs/getting-started/ace-windows.md#ollama) |
| **OpenAI-compatible server** | Para usar LM Studio ou outro servidor compatível no mesmo computador. [Configurar servidor local](docs/getting-started/ace-windows.md#lm-studio-ou-outro-servidor-local) |
| **Cloudflare Workers AI** | Informe o ID da conta e o token no aplicativo e escolha o modelo. O texto da conversa é enviado à Cloudflare. [Configurar Cloudflare](docs/getting-started/ace-windows.md#cloudflare-workers-ai) |

Conteúdo novo na web requer internet. Para consultar o que já foi salvo, abra **Web Research → Search saved sources**. Você também pode corrigir registros em **Memory Facts** e consultar as relações em **Knowledge Graph**. [Guia de recursos do A.C.E.](docs/user-guide/ace.md)

## Contribuir

O [guia de contribuição](CONTRIBUTING.md) mostra como executar o código, testar alterações e preparar o instalador. Abra uma issue neste repositório para relatar um erro ou propor uma mudança.

## Origem e licença

O A.C.E. deriva do OpenJarvis, desenvolvido pelo Hazy Research e pelo Scaling Intelligence Lab de Stanford. O código é distribuído sob a licença Apache 2.0. [Código original](https://github.com/open-jarvis/OpenJarvis) · [Artigo](https://arxiv.org/abs/2605.17172) · [Licença](LICENSE)
