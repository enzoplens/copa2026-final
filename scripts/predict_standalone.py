"""
Copa do Mundo 2026 — Módulo de Predição
========================================
Carrega o modelo treinado e expõe a função predict_match().

Uso:
    from predict import predict_match
    result = predict_match('Brazil', 'France', phase='sf')
    # → {'win1': 57.3, 'draw': 24.1, 'win2': 18.6, 'score1': 2, 'score2': 1}
"""

import pickle, json
import pandas as pd

# ── Ranking FIFA aproximado para Copa 2026 ──
RANK_2026 = {
    'Argentina':1, 'France':2, 'England':3, 'Spain':4, 'Brazil':5,
    'Portugal':6, 'Netherlands':7, 'Belgium':8, 'Germany':9,
    'Croatia':12, 'Morocco':14, 'Colombia':15, 'USA':13, 'Mexico':16,
    'Uruguay':17, 'Japan':18, 'Senegal':19, 'Switzerland':20,
    'South Korea':22, 'Iran':22, 'Austria':24, 'Australia':25,
    'Sweden':25, 'Turkey':28, 'Tunisia':28, 'Norway':30,
    'Paraguay':30, 'Algeria':35, 'Czech Republic':37, 'Scotland':39,
    'Canada':41, 'Ecuador':44, 'Cameroon':43, 'Saudi Arabia':56,
    'Ghana':60, 'South Africa':65, 'Iraq':70, 'Panama':70,
    'DR Congo':72, 'Uzbekistan':75, 'Cape Verde':80, 'Jordan':88,
    'Haiti':95, 'New Zealand':96, 'Qatar':50, 'Curacao':85,
    'Serbia':21, 'Ivory Coast':52, 'Egypt':34,
}

PHASE_MAP = {'group':0, 'r16':1, 'qf':2, 'sf':3, '3rd':3, 'final':4}


def load_model(model_path='../models/wc_model.pkl',
               stats_path='../data/wc_stats.json'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(stats_path) as f:
        stats = json.load(f)
    return model, stats


def predict_match(team1: str, team2: str, phase: str = 'group',
                  model=None, stats=None):
    """
    Retorna probabilidades e placar mais provável para team1 vs team2.

    Parâmetros
    ----------
    team1, team2 : str  — nomes em inglês conforme RANK_2026
    phase        : str  — 'group' | 'r16' | 'qf' | 'sf' | 'final'

    Retorno
    -------
    dict com chaves: win1, draw, win2, score1, score2
    """
    if model is None or stats is None:
        model, stats = load_model()

    FEATURES     = stats['features']
    form_rate    = stats['form_rate']
    avg_goals    = stats['avg_goals']
    titles       = stats['titles']
    team_games   = stats['team_games']

    r1  = RANK_2026.get(team1, 50)
    r2  = RANK_2026.get(team2, 50)
    ph  = PHASE_MAP.get(phase, 0)
    f1  = form_rate.get(team1, 0.38)
    f2  = form_rate.get(team2, 0.38)
    g1  = avg_goals.get(team1, 1.2)
    g2  = avg_goals.get(team2, 1.2)
    t1  = titles.get(team1, 0)
    t2  = titles.get(team2, 0)
    e1  = team_games.get(team1, 4)
    e2  = team_games.get(team2, 4)

    row = pd.DataFrame([[
        r1 - r2,
        r2 / (r1 + 0.01),
        f1 - f2,
        g1 - g2,
        t1 - t2,
        e1 - e2,
        ph
    ]], columns=FEATURES)

    probs  = model.predict_proba(row)[0]
    prob_d = dict(zip(model.classes_, probs))

    win1 = round(prob_d.get(1, 0) * 100, 1)
    draw = round(prob_d.get(0, 0) * 100, 1)
    win2 = round(prob_d.get(2, 0) * 100, 1)

    # Placar mais provável ponderado pela probabilidade de vitória
    factor1 = (win1 / 100) * 1.5
    factor2 = (win2 / 100) * 1.5
    total   = max(g1 * factor1 + g2 * factor2, 0.01)
    score1  = int(round(g1 * factor1 / total * (g1 + g2)))
    score2  = int(round(g2 * factor2 / total * (g1 + g2)))

    return {
        'win1':   win1,
        'draw':   draw,
        'win2':   win2,
        'score1': score1,
        'score2': score2,
    }


if __name__ == '__main__':
    model, stats = load_model()
    fixtures = [
        ('Brazil',    'Morocco',  'group'),
        ('Argentina', 'France',   'final'),
        ('Spain',     'England',  'sf'),
        ('Germany',   'France',   'qf'),
        ('Brazil',    'Argentina','sf'),
    ]
    print(f"\n{'─'*60}")
    print(f"{'Partida':<30} {'T1':>6}  {'Emp':>6}  {'T2':>6}  {'Placar'}")
    print(f"{'─'*60}")
    for t1, t2, ph in fixtures:
        p = predict_match(t1, t2, ph, model, stats)
        print(f"{t1} vs {t2} ({ph}){'':<5}"
              f"{p['win1']:>6}%  {p['draw']:>6}%  {p['win2']:>6}%  "
              f"{p['score1']}-{p['score2']}")
