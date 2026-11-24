# Instalar e configurar o A.C.E. no Windows

## Instalar ou atualizar

1. Baixe e abra [`Instalar-A.C.E.exe`](https://github.com/renato-chagas/A.C.E/releases/latest/download/Instalar-A.C.E.exe) no Windows de 64 bits. Clique em **Instalar** para usar a pasta sugerida ou escolha outra. O instalador inclui Python e WebView2. Mantenha a opção de abrir o A.C.E. marcada ao concluir.
2. Na primeira abertura, selecione **Configure model** se quiser conversar. Você pode escolher **Continue with local services** para usar memória e fontes salvas antes de configurar um modelo.

Nas próximas vezes, abra o A.C.E. pelo menu Iniciar ou pelo [launcher opcional](https://github.com/renato-chagas/A.C.E/releases/latest/download/Iniciar-ACE.cmd). Para atualizar, clique no ícone perto do relógio e escolha **Sair do A.C.E.**. Rode o novo instalador mantendo a pasta da instalação existente.

## Ollama

1. Instale e abra o Ollama. No PowerShell, rode `ollama list` para ver os modelos disponíveis no computador. [Instalador do Ollama para Windows](https://ollama.com/download/windows)
2. Se precisar baixar um modelo para o primeiro teste, rode `ollama pull qwen3:0.6b` com internet. [Modelo Qwen3 0.6B](https://ollama.com/library/qwen3:0.6b)
3. No A.C.E., abra **Configure model → Local Ollama**. Informe o nome completo que aparece em `ollama list`, como `qwen3:0.6b`, e clique em **Use this model and continue**.
4. Envie uma mensagem no chat. Com o modelo baixado e o Ollama disponível, a conversa pode funcionar sem internet.

## LM Studio ou outro servidor local

1. Abra o LM Studio, carregue um modelo e ligue o servidor em **Developer → Start server**. Use um endereço local, como `http://127.0.0.1:1234`. [Documentação do servidor LM Studio](https://lmstudio.ai/docs/developer/core/server)
2. No A.C.E., abra **Configure model → OpenAI-compatible server** e selecione **LM Studio**. Informe a URL compatível, como `http://127.0.0.1:1234/v1`, e o identificador de modelo exibido no LM Studio. Clique em **Save and connect**.
3. Para outro servidor compatível, selecione o tipo correspondente e use o endereço local e o identificador que ele anuncia. Mantenha o servidor e o modelo disponíveis enquanto usa o chat.

## Cloudflare Workers AI

Escolha esta opção para usar o Gemma 4 online. O A.C.E. envia o texto da conversa atual à Cloudflare. Consulte as condições de uso e a cota da sua conta no painel do provedor. [Documentação do Workers AI](https://developers.cloudflare.com/workers-ai/)

1. No painel da Cloudflare, entre na conta desejada e abra **Workers AI → Use REST API**. Copie o **Account ID**. Crie um token do Workers AI e guarde-o separadamente do ID. [Guia oficial de credenciais](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
2. Instale o A.C.E. e feche qualquer instância aberta pelo ícone perto do relógio em **Sair do A.C.E.**. Abra o PowerShell na pasta que contém o [launcher `Iniciar-ACE.cmd`](https://github.com/renato-chagas/A.C.E/releases/latest/download/Iniciar-ACE.cmd).
3. Execute estas linhas. Quando aparecer a primeira pergunta, cole o ID da conta. Na segunda, cole o token e pressione Enter, mesmo que o texto fique oculto.

   ```powershell
   $env:CLOUDFLARE_ACCOUNT_ID = (Read-Host 'ID da conta Cloudflare').Trim()
   $env:CLOUDFLARE_API_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Token do Workers AI' -AsSecureString)).Password.Trim()
   & '.\Iniciar-ACE.cmd'
   Remove-Item Env:CLOUDFLARE_API_TOKEN
   ```

4. No A.C.E., abra **Configure model → Cloudflare · Gemma 4 → Connect to Cloudflare**. Teste com uma mensagem curta em **Chat**.

Se já estiver usando outro modelo, selecione Cloudflare em **Settings → Inference source**, clique em **Save Cloudflare for next launch** e reinicie pelo PowerShell conforme os passos acima. O aplicativo lê as credenciais ao abrir. Para usar Cloudflare em outra sessão, execute as linhas de abertura novamente.

## Se algo não abrir

- **O chat local está indisponível.** Confira se o Ollama ou o LM Studio está aberto e se o modelo escolhido continua disponível. Use **Retry saved model** quando aparecer.
- **O A.C.E. pede as variáveis da Cloudflare.** Feche-o pelo ícone perto do relógio e execute de novo o bloco no mesmo PowerShell em que você informou o ID e o token.
- **A Cloudflare rejeita a chave.** Verifique se o token pertence à conta identificada pelo Account ID e se tem acesso ao Workers AI. [Guia oficial de credenciais](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
- **A porta 8000 já está em uso.** Se o servidor existente pertence ao mesmo A.C.E. e ao mesmo perfil, o aplicativo pode reaproveitá-lo. Se a mensagem indicar outro serviço, identifique-o antes de fechar qualquer processo.
