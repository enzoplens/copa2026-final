# Copa 2026 Analytics — Projeto completo

Este é o projeto fundido e completo: o pipeline de Machine Learning, a
automação que atualiza os dados a cada hora via GitHub Actions, e o
site que lê esses dados dinamicamente.

> **Nota sobre esta versão:** os dois zips anteriores (`copa2026.zip` e
> `copa2026-live.zip`) tinham um `train.py` cada, treinados em datasets
> de tamanhos diferentes (210 partidas vs 125). O modelo de 210 partidas
> (71.0% de acurácia) é o que está empacotado aqui — o `wc_model.pkl` e
> `wc_stats.json` foram verificados como compatíveis com o `update.py`
> e `predict_standalone.py` (mesmas 7 features, mesma estrutura de
> Pipeline do scikit-learn) antes da fusão.

```
copa2026-final/
├── .github/workflows/update.yml     ← roda a cada hora no GitHub
├── scripts/
│   ├── train.py                     ← treina o modelo do zero (210 partidas)
│   ├── update.py                    ← busca API + roda modelo + escreve live.json
│   ├── seed_schedule.py             ← agenda fixa usada só na 1ª execução
│   └── predict_standalone.py        ← módulo de predição isolado, sem depender do pipeline do GitHub Actions
├── models/wc_model.pkl              ← modelo treinado (Logistic Regression, 71.0%, 210 jogos)
├── data/
│   ├── wc_stats.json                ← métricas e stats derivadas do treino
│   ├── live.json                    ← gerado por update.py (o site lê este arquivo)
│   └── last_updated.txt
└── site/index.html                  ← frontend que faz fetch em data/live.json
```

### Sobre `predict_standalone.py`

Diferente de `update.py` (que é parte do pipeline automático e espera
rodar dentro do GitHub Actions), este módulo serve para você testar
previsões pontuais direto no terminal, sem depender de API externa ou
do workflow:

```bash
cd scripts
python3 predict_standalone.py
```

```
Partida                            T1     Emp      T2  Placar
Brazil vs Morocco (group)       64.8%    16.6%    18.6%  2-0
Argentina vs France (final)     31.6%    27.2%    41.2%  1-3
```

Ou import direto:
```python
from predict_standalone import predict_match, load_model
model, stats = load_model()
predict_match('Brazil', 'France', phase='sf', model=model, stats=stats)
```

---

## ⚠️ Importante: não abra index.html com duplo clique

Testei isso e confirmei o problema: se você abrir `site/index.html` direto
do seu computador (URL começando com `file://`), o navegador **bloqueia**
o `fetch()` para `data/live.json` por política de segurança (CORS), mesmo
que o arquivo exista do lado do JSON. O site vai mostrar corretamente a
tela de erro ("Não foi possível carregar dados ao vivo") — isso é o
comportamento esperado nessa situação, não um bug.

Para o fetch funcionar, o site precisa ser **servido**, não aberto como
arquivo. Duas formas simples:

**Localmente, para testar:**
```bash
cd copa2026-live/site
python3 -m http.server 8000
# abra http://localhost:8000/index.html
```

**Em produção, via GitHub Pages (recomendado):**
1. No repositório → Settings → Pages
2. Source: branch `main`, pasta `/site` (ou mova `data/` para dentro de `site/` antes)
3. O GitHub te dá uma URL tipo `https://SEU_USUARIO.github.io/copa2026/`

Se preferir manter `data/` fora de `site/`, edite a constante no topo do
`<script>` em `index.html`:
```js
const LIVE_JSON_URL = 'https://raw.githubusercontent.com/SEU_USUARIO/copa2026/main/data/live.json';
```

---

## Setup do zero

```bash
# 1. Treina o modelo (gera models/wc_model.pkl e data/wc_stats.json)
pip install scikit-learn pandas numpy requests
python scripts/train.py

# 2. Testa o pipeline de atualização localmente (sem token, usa seed_schedule.py)
python scripts/update.py
# → gera data/live.json com a agenda fixa + previsões ML

# 3. Sobe pro GitHub
git init && git add . && git commit -m "setup copa2026-live"
git remote add origin https://github.com/SEU_USUARIO/copa2026.git
git push -u origin main
```

Depois, no GitHub:
1. **Settings → Secrets and variables → Actions → New repository secret**
   Nome: `FOOTBALL_API_TOKEN` · Valor: seu token de football-data.org
2. **Actions** → workflow "Atualizar dados Copa 2026" → **Run workflow** (teste manual)
3. A partir daí ele roda sozinho, todo início de hora.

---

## O que o `update.py` faz, passo a passo

1. Tenta buscar `https://api.football-data.org/v4/competitions/WC/matches`
   com o token do secret.
2. **Se a API responder:** processa os jogos reais (placar, status, fase).
3. **Se a API falhar ou o token não estiver configurado:** cai no
   `data/live.json` da execução anterior; se esse também não existir
   (primeira execução), usa `seed_schedule.py` — uma agenda fixa com os
   4 primeiros jogos da Copa (já com placar real) e mais 10 jogos
   programados. Isso evita que o site fique com `matches: []` antes do
   primeiro fetch bem-sucedido — esse era exatamente o bug que apareceu
   no teste inicial deste pipeline, corrigido na função `load_existing()`.
4. Roda o modelo ML (`predict_match()`) em todo jogo com `status: sched`.
5. Simula 5.000 torneios para estimar probabilidade de título por seleção.
6. Escreve `data/live.json` com tudo isso + timestamp.

---

## Testado com

- `python3 -m http.server` + Playwright (Chromium headless) para confirmar
  que os dados realmente renderizam no DOM, não só que o JSON é válido.
- Cenário de falha real (sem `live.json`): confirma que a UI mostra erro
  honesto em vez de tela vazia ou dado fictício.
- `node --check` na sintaxe JS extraída do HTML.

Não testado: a chamada real à API football-data.org (não tenho token).
O `NAME_MAP` em `update.py` foi escrito com base na documentação da API,
mas pode haver nomes de seleção que a API retorna e que ainda não estão
mapeados — se isso acontecer, o time aparece com ranking padrão (50) em
vez do ranking real. Verifique os logs do GitHub Actions após a primeira
execução com token de verdade.
