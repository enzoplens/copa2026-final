"""
scripts/update.py
=================
Script principal do GitHub Actions.

Fluxo:
  1. Busca resultados reais da API football-data.org
  2. Carrega o modelo ML treinado (scikit-learn)
  3. Roda previsões em todos os jogos ainda não realizados
  4. Calcula probabilidades de título via simulação
  5. Escreve data/live.json  ← lido pelo site
  6. Escreve data/last_updated.txt

Rodar localmente:
  export FOOTBALL_API_TOKEN=seu_token_aqui
  python scripts/update.py
"""

import os, json, pickle, requests, datetime, warnings
import pandas as pd
import numpy as np
from seed_schedule import SEED_MATCHES

warnings.filterwarnings('ignore')

# ─── Configuração ───────────────────────────────────────────
API_TOKEN   = os.environ.get('FOOTBALL_API_TOKEN', '')
API_BASE    = 'https://api.football-data.org/v4'
COMPETITION = 'WC'          # código da Copa do Mundo na API
MODEL_PATH  = 'models/wc_model.pkl'
STATS_PATH  = 'data/wc_stats.json'
OUTPUT_PATH = 'data/live.json'
UPDATED_PATH = 'data/last_updated.txt'

# ─── Ranking FIFA 2026 ──────────────────────────────────────
RANK_2026 = {
    'Argentina':1,'France':2,'England':3,'Spain':4,'Brazil':5,
    'Portugal':6,'Netherlands':7,'Belgium':8,'Germany':9,
    'Croatia':12,'Morocco':14,'Colombia':15,'USA':13,'Mexico':16,
    'Uruguay':17,'Japan':18,'Senegal':19,'Switzerland':20,
    'South Korea':22,'Iran':22,'Austria':24,'Australia':25,
    'Turkey':28,'Tunisia':28,'Norway':30,'Paraguay':30,
    'Algeria':35,'Czech Republic':37,'Scotland':39,'Canada':41,
    'Ecuador':44,'Cameroon':43,'Saudi Arabia':56,'Ghana':60,
    'South Africa':65,'Iraq':70,'Panama':70,'DR Congo':72,
    'Uzbekistan':75,'Cape Verde':80,'Jordan':88,'Haiti':95,
    'New Zealand':96,'Qatar':50,'Curacao':85,'Serbia':21,
    'Ivory Coast':52,'Egypt':34,'Sweden':25,'Ghana':60,
}

# Mapa nome API → nome no nosso modelo
NAME_MAP = {
    'United States': 'USA',
    'Korea Republic': 'South Korea',
    'Côte d\'Ivoire': 'Ivory Coast',
    'IR Iran': 'Iran',
    'DR Congo': 'DR Congo',
    'Cape Verde Islands': 'Cape Verde',
    'Trinidad and Tobago': 'Trinidad & Tobago',
}

PHASE_MAP = {'GROUP_STAGE':0,'ROUND_OF_16':1,'QUARTER_FINALS':2,
             'SEMI_FINALS':3,'THIRD_PLACE':3,'FINAL':4}

# ─── Flags por nome ─────────────────────────────────────────
FLAGS = {
    'Argentina':'🇦🇷','France':'🇫🇷','England':'🏴󠁧󠁢󠁥󠁮󠁧󠁿','Spain':'🇪🇸','Brazil':'🇧🇷',
    'Portugal':'🇵🇹','Netherlands':'🇳🇱','Belgium':'🇧🇪','Germany':'🇩🇪',
    'Croatia':'🇭🇷','Morocco':'🇲🇦','Colombia':'🇨🇴','USA':'🇺🇸','Mexico':'🇲🇽',
    'Uruguay':'🇺🇾','Japan':'🇯🇵','Senegal':'🇸🇳','Switzerland':'🇨🇭',
    'South Korea':'🇰🇷','Iran':'🇮🇷','Austria':'🇦🇹','Australia':'🇦🇺',
    'Turkey':'🇹🇷','Tunisia':'🇹🇳','Norway':'🇳🇴','Paraguay':'🇵🇾',
    'Algeria':'🇩🇿','Czech Republic':'🇨🇿','Scotland':'🏴󠁧󠁢󠁳󠁣󠁴󠁿','Canada':'🇨🇦',
    'Ecuador':'🇪🇨','Cameroon':'🇨🇲','Saudi Arabia':'🇸🇦','Ghana':'🇬🇭',
    'South Africa':'🇿🇦','Iraq':'🇮🇶','Panama':'🇵🇦','DR Congo':'🇨🇩',
    'Uzbekistan':'🇺🇿','Cape Verde':'🇨🇻','Jordan':'🇯🇴','Haiti':'🇭🇹',
    'New Zealand':'🇳🇿','Qatar':'🇶🇦','Curacao':'🇨🇼','Serbia':'🇷🇸',
    'Ivory Coast':'🇨🇮','Egypt':'🇪🇬','Sweden':'🇸🇪','Belgium':'🇧🇪',
}

# ─── Carrega modelo e stats ──────────────────────────────────
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(STATS_PATH) as f:
        stats = json.load(f)
    return model, stats

# ─── Predição para uma partida ───────────────────────────────
def predict_match(t1, t2, phase_enc, model, stats):
    FEATURES   = stats['features']
    form_rate  = stats['form_rate']
    avg_goals  = stats['avg_goals']
    titles     = stats['titles']
    team_games = stats['team_games']

    r1 = RANK_2026.get(t1, 50);  r2 = RANK_2026.get(t2, 50)
    f1 = form_rate.get(t1, .38); f2 = form_rate.get(t2, .38)
    g1 = avg_goals.get(t1, 1.2); g2 = avg_goals.get(t2, 1.2)
    t1v = titles.get(t1, 0);     t2v = titles.get(t2, 0)
    e1 = team_games.get(t1, 4);  e2 = team_games.get(t2, 4)

    row = pd.DataFrame([[
        r1-r2, r2/(r1+.01), f1-f2, g1-g2, t1v-t2v, e1-e2, phase_enc
    ]], columns=FEATURES)

    probs  = model.predict_proba(row)[0]
    pd_    = dict(zip(model.classes_, probs))
    win1   = round(pd_.get(1, 0)*100, 1)
    draw   = round(pd_.get(0, 0)*100, 1)
    win2   = round(pd_.get(2, 0)*100, 1)

    fac1 = (win1/100)*1.5; fac2 = (win2/100)*1.5
    tot  = max(g1*fac1 + g2*fac2, .01)
    s1   = int(round(g1*fac1/tot*(g1+g2)))
    s2   = int(round(g2*fac2/tot*(g1+g2)))

    return {'win1':win1, 'draw':draw, 'win2':win2, 'score1':s1, 'score2':s2}

# ─── Busca resultados na API ─────────────────────────────────
def fetch_matches():
    """
    Retorna lista de partidas da Copa 2026 com resultados reais.
    Se a API falhar (sem token, rate limit), retorna lista vazia
    e o script usa os dados já existentes.
    """
    if not API_TOKEN:
        print("⚠️  FOOTBALL_API_TOKEN não definido — usando dados locais.")
        return []

    headers = {'X-Auth-Token': API_TOKEN}
    url     = f'{API_BASE}/competitions/{COMPETITION}/matches'

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        raw  = resp.json().get('matches', [])
        print(f"✅  API: {len(raw)} partidas recebidas")
        return raw
    except requests.HTTPError as e:
        print(f"⚠️  Erro HTTP da API: {e}")
        return []
    except Exception as e:
        print(f"⚠️  Falha na API: {e}")
        return []

# ─── Processa partida da API → formato padrão ────────────────
def safe_team_name(team_obj):
    """
    Extrai o nome do time com segurança. Partidas de fases futuras
    ainda não definidas (ex: 'Vencedor do Grupo A vs Vencedor do
    Grupo B') costumam vir com homeTeam/awayTeam como null ou com
    name ausente. Sem este tratamento, isso quebra tanto aqui
    quanto no frontend (matchCard chama .split() no nome do time).
    """
    if not team_obj or not isinstance(team_obj, dict):
        return 'A definir'
    name = team_obj.get('name')
    if not name:
        return 'A definir'
    return NAME_MAP.get(name, name)


def process_api_match(m):
    home = safe_team_name(m.get('homeTeam'))
    away = safe_team_name(m.get('awayTeam'))

    status = m['status']           # FINISHED | SCHEDULED | IN_PLAY | PAUSED
    stage  = m.get('stage', 'GROUP_STAGE')
    group  = m.get('group') or stage

    score = m.get('score', {})
    ft    = score.get('fullTime', {})
    s1    = ft.get('home')         # None se não encerrado
    s2    = ft.get('away')

    utc   = m.get('utcDate', '')
    try:
        dt   = datetime.datetime.fromisoformat(utc.replace('Z','+00:00'))
        date = dt.strftime('%d %b')
        time = dt.strftime('%H:%M') + ' UTC'
    except:
        date = '—'; time = '—'

    return {
        'id':     m.get('id'),
        't1':     home,
        't2':     away,
        'f1':     FLAGS.get(home, '🏳'),
        'f2':     FLAGS.get(away, '🏳'),
        'group':  group,
        'stage':  stage,
        'date':   date,
        'time':   time,
        's1':     s1,
        's2':     s2,
        'status': 'done' if status == 'FINISHED' else
                  'live' if status in ('IN_PLAY','PAUSED') else 'sched',
    }

# ─── Simula torneio → prob de título ────────────────────────
def simulate_tournament(model, stats, n=5000):
    """
    Simula n torneios completos e conta quantas vezes cada
    seleção vence para estimar probabilidade de título.
    """
    CONTENDERS = list({k for k in RANK_2026 if RANK_2026[k] <= 35})
    titles_count = {t: 0 for t in CONTENDERS}

    for _ in range(n):
        # Sorteio simples: 16 seleções no mata-mata
        pool = np.random.choice(CONTENDERS, size=16, replace=False).tolist()
        survivors = pool

        for phase_enc in [1, 2, 3, 4]:   # oitavas, quartas, semi, final
            next_round = []
            pairs = [(survivors[i], survivors[i+1])
                     for i in range(0, len(survivors), 2)]
            for t1, t2 in pairs:
                p = predict_match(t1, t2, phase_enc, model, stats)
                rand = np.random.random() * 100
                if rand < p['win1']:
                    next_round.append(t1)
                elif rand < p['win1'] + p['draw']:
                    # No empate, decide por pênaltis (50/50)
                    next_round.append(t1 if np.random.random() > 0.5 else t2)
                else:
                    next_round.append(t2)
            survivors = next_round

        if survivors:
            titles_count[survivors[0]] = titles_count.get(survivors[0], 0) + 1

    # Normaliza para %
    total = sum(titles_count.values()) or 1
    probs = {k: round(v/total*100, 1)
             for k, v in sorted(titles_count.items(), key=lambda x:-x[1])
             if v > 0}
    return probs

# ─── Carrega live.json existente (fallback) ──────────────────
def load_existing():
    """
    Tenta carregar o live.json da execução anterior. Se não existir
    ainda (primeira execução do workflow) ou estiver corrompido,
    cai para a agenda fixa em seed_schedule.py em vez de uma lista
    vazia — sem isso, a primeira rodada do GitHub Actions publica
    matches: [] e o site fica sem nenhum jogo até a API responder
    com sucesso pela primeira vez.
    """
    try:
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
            if data.get('matches'):
                return data
            print("ℹ️   live.json existe mas está sem partidas — usando seed.")
    except (FileNotFoundError, json.JSONDecodeError):
        print("ℹ️   live.json ainda não existe (primeira execução) — usando seed.")

    return {
        'matches': [dict(m) for m in SEED_MATCHES],
        'titles': {},
        'last_updated': '—',
        'model': {},
    }

# ─── Main ────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("Copa 2026 — Pipeline de atualização")
    print("=" * 55)

    # 1. Carrega modelo ML
    try:
        model, stats = load_model()
        print(f"✅  Modelo: {stats['best_model']} ({stats['accuracy']}%)")
    except FileNotFoundError:
        print("❌  Modelo não encontrado. Rode scripts/train.py primeiro.")
        raise

    # 2. Busca resultados da API
    raw_matches = fetch_matches()

    # 3. Se API retornou dados, processa; caso contrário usa existing
    existing = load_existing()

    if raw_matches:
        matches = [process_api_match(m) for m in raw_matches]

        # Adiciona previsões ML nos jogos não encerrados, exceto
        # confrontos futuros do mata-mata cujos times ainda não
        # foram definidos (ex: "Vencedor do Grupo A vs ...").
        # Prever isso geraria probabilidades sem sentido nenhum.
        for m in matches:
            if m['t1'] == 'A definir' or m['t2'] == 'A definir':
                continue
            ph = PHASE_MAP.get(m['stage'], 0)
            pred = predict_match(m['t1'], m['t2'], ph, model, stats)
            m['w1']     = pred['win1']
            m['draw']   = pred['draw']
            m['w2']     = pred['win2']
            m['ps1']    = pred['score1']
            m['ps2']    = pred['score2']

        print(f"✅  {len(matches)} partidas processadas")
    else:
        # API falhou: preserva placares reais existentes, só atualiza previsões
        matches = existing.get('matches', [])
        print(f"ℹ️   Usando {len(matches)} partidas do cache local")
        for m in matches:
            if m['status'] == 'sched' and m['t1'] != 'A definir' and m['t2'] != 'A definir':
                ph   = PHASE_MAP.get(m.get('stage','GROUP_STAGE'), 0)
                pred = predict_match(m['t1'], m['t2'], ph, model, stats)
                m['w1']  = pred['win1']
                m['draw'] = pred['draw']
                m['w2']  = pred['win2']
                m['ps1'] = pred['score1']
                m['ps2'] = pred['score2']

    # 4. Simula torneio para probabilidades de título
    print("🎲  Simulando torneio (5.000 iterações)...")
    title_probs = simulate_tournament(model, stats, n=5000)
    print(f"✅  Top 3: " + ", ".join(
        f"{k} {v}%" for k,v in list(title_probs.items())[:3]))

    # 5. Estatísticas de grupo derivadas dos resultados reais
    group_stats = {}
    for m in matches:
        if m['status'] == 'done' and m.get('s1') is not None:
            for side, team, gf, ga in [
                ('home', m['t1'], m['s1'], m['s2']),
                ('away', m['t2'], m['s2'], m['s1'])
            ]:
                grp = m['group']
                if grp not in group_stats:
                    group_stats[grp] = {}
                if team not in group_stats[grp]:
                    group_stats[grp][team] = {
                        'flag': FLAGS.get(team,'🏳'),
                        'p':0,'w':0,'d':0,'l':0,'gf':0,'ga':0,'pts':0
                    }
                s = group_stats[grp][team]
                s['p']  += 1
                s['gf'] += gf
                s['ga'] += ga
                if gf > ga:   s['w']+=1; s['pts']+=3
                elif gf==ga:  s['d']+=1; s['pts']+=1
                else:         s['l']+=1

    # 6. Monta o JSON final
    now     = datetime.datetime.utcnow().strftime('%d/%m/%Y %H:%M UTC')
    payload = {
        'last_updated': now,
        'model': {
            'name':     stats['best_model'],
            'accuracy': stats['accuracy'],
            'n_train':  stats['n_matches'],
        },
        'matches':      matches,
        'titles':       title_probs,
        'group_stats':  group_stats,
    }

    # 7. Salva
    os.makedirs('data', exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(UPDATED_PATH, 'w') as f:
        f.write(now)

    print(f"\n✅  {OUTPUT_PATH} atualizado — {now}")
    print("=" * 55)

if __name__ == '__main__':
    main()
