"""
scripts/train.py
================
Treina o modelo ML e salva os artefatos necessários para update.py.

⚠️  ATENÇÃO — este arquivo treina com um dataset de 125 partidas (~64-68%
de acurácia). O models/wc_model.pkl já incluído neste projeto foi treinado
separadamente com 210 partidas (71.0% de acurácia) e é estritamente melhor.

Rodar este script vai SOBRESCREVER esse modelo melhor pelo mais fraco.
Só rode se você for expandir o MATCHES_RAW abaixo com mais partidas reais
(o objetivo natural seria superar 210 jogos, não regredir para 125).

  pip install scikit-learn pandas numpy
  python scripts/train.py
"""

import pandas as pd, numpy as np, json, pickle, os, warnings
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

MATCHES_RAW = [
    ("Qatar","Ecuador",0,2,"group",2022,50,44),("England","Iran",6,2,"group",2022,5,21),
    ("USA","Wales",1,1,"group",2022,13,19),("Argentina","Saudi Arabia",1,2,"group",2022,3,51),
    ("France","Australia",4,1,"group",2022,4,38),("Germany","Japan",1,2,"group",2022,11,24),
    ("Spain","Costa Rica",7,0,"group",2022,7,31),("Belgium","Canada",1,0,"group",2022,2,41),
    ("Brazil","Serbia",2,0,"group",2022,1,21),("France","Denmark",2,1,"group",2022,4,10),
    ("Argentina","Mexico",2,0,"group",2022,3,13),("Belgium","Morocco",0,2,"group",2022,2,22),
    ("Croatia","Canada",4,1,"group",2022,12,41),("Spain","Germany",1,1,"group",2022,7,11),
    ("Brazil","Switzerland",1,0,"group",2022,1,15),("Portugal","Uruguay",2,0,"group",2022,9,14),
    ("Poland","Argentina",0,2,"group",2022,26,3),("Japan","Spain",2,1,"group",2022,24,7),
    ("South Korea","Portugal",2,1,"group",2022,28,9),("Cameroon","Brazil",1,0,"group",2022,43,1),
    ("Netherlands","USA",3,1,"r16",2022,8,13),("Argentina","Australia",2,1,"r16",2022,3,38),
    ("France","Poland",3,1,"r16",2022,4,26),("England","Senegal",3,0,"r16",2022,5,18),
    ("Brazil","South Korea",4,1,"r16",2022,1,28),("Morocco","Spain",0,0,"r16",2022,22,7),
    ("Portugal","Switzerland",6,1,"r16",2022,9,15),("Netherlands","Argentina",2,2,"qf",2022,8,3),
    ("Croatia","Brazil",1,1,"qf",2022,12,1),("Morocco","Portugal",1,0,"qf",2022,22,9),
    ("England","France",1,2,"qf",2022,5,4),("Argentina","Croatia",3,0,"sf",2022,3,12),
    ("France","Morocco",2,0,"sf",2022,4,22),("Argentina","France",3,3,"final",2022,3,4),
    ("Portugal","Spain",3,3,"group",2018,4,10),("Germany","Mexico",0,1,"group",2018,1,16),
    ("Brazil","Switzerland",1,1,"group",2018,2,6),("Belgium","Panama",3,0,"group",2018,3,55),
    ("Colombia","Japan",1,2,"group",2018,16,60),("Argentina","Croatia",0,3,"group",2018,5,20),
    ("Brazil","Costa Rica",2,0,"group",2018,2,22),("Germany","Sweden",2,1,"group",2018,1,24),
    ("South Korea","Germany",2,0,"group",2018,57,1),("France","Argentina",4,3,"r16",2018,7,5),
    ("Uruguay","Portugal",2,1,"r16",2018,17,4),("Croatia","Denmark",1,1,"r16",2018,20,12),
    ("Brazil","Mexico",2,0,"r16",2018,2,16),("Belgium","Japan",3,2,"r16",2018,3,60),
    ("Uruguay","France",0,2,"qf",2018,17,7),("Brazil","Belgium",1,2,"qf",2018,2,3),
    ("Russia","Croatia",2,2,"qf",2018,70,20),("Sweden","England",0,2,"qf",2018,24,13),
    ("France","Belgium",1,0,"sf",2018,7,3),("Croatia","England",2,1,"sf",2018,20,13),
    ("France","Croatia",4,2,"final",2018,7,20),
    ("Brazil","Croatia",3,1,"group",2014,4,18),("Spain","Netherlands",1,5,"group",2014,1,15),
    ("Colombia","Greece",3,0,"group",2014,5,12),("Uruguay","Costa Rica",1,3,"group",2014,6,28),
    ("England","Italy",1,2,"group",2014,10,9),("Germany","Portugal",4,0,"group",2014,2,4),
    ("Spain","Chile",0,2,"group",2014,1,14),("Germany","Algeria",2,1,"r16",2014,2,26),
    ("Argentina","Switzerland",1,0,"r16",2014,8,7),("Belgium","USA",2,1,"r16",2014,11,13),
    ("Colombia","Uruguay",2,0,"r16",2014,5,6),("France","Nigeria",2,0,"r16",2014,3,44),
    ("Brazil","Chile",1,1,"r16",2014,4,14),("Brazil","Colombia",2,1,"qf",2014,4,5),
    ("France","Germany",0,1,"qf",2014,3,2),("Argentina","Belgium",1,0,"qf",2014,8,11),
    ("Brazil","Germany",1,7,"sf",2014,4,2),("Germany","Argentina",1,0,"final",2014,2,8),
    ("Germany","Australia",4,0,"group",2010,6,96),("Netherlands","Denmark",2,0,"group",2010,4,34),
    ("Brazil","North Korea",2,1,"group",2010,1,105),("Spain","Switzerland",0,1,"group",2010,2,25),
    ("Brazil","Ivory Coast",3,1,"group",2010,1,27),("Argentina","Mexico",3,1,"r16",2010,7,17),
    ("Germany","England",4,1,"r16",2010,6,8),("Brazil","Chile",3,0,"r16",2010,1,18),
    ("Spain","Portugal",1,0,"r16",2010,2,8),("Uruguay","South Korea",2,1,"r16",2010,16,47),
    ("Netherlands","Brazil",2,1,"qf",2010,4,1),("Argentina","Germany",0,4,"qf",2010,7,6),
    ("Paraguay","Spain",0,1,"qf",2010,31,2),("Netherlands","Uruguay",3,2,"sf",2010,4,16),
    ("Germany","Spain",0,1,"sf",2010,6,2),("Netherlands","Spain",0,1,"final",2010,4,2),
    ("Germany","Costa Rica",4,2,"group",2006,22,21),("Spain","Ukraine",4,0,"group",2006,5,49),
    ("Brazil","Croatia",1,0,"group",2006,1,20),("Argentina","Serbia & Montenegro",6,0,"group",2006,4,38),
    ("Germany","Ecuador",3,0,"r16",2006,22,37),("Brazil","Ghana",3,0,"r16",2006,1,48),
    ("Spain","France",1,3,"r16",2006,5,5),("Argentina","Mexico",2,1,"r16",2006,4,7),
    ("Germany","Argentina",1,1,"qf",2006,22,4),("Brazil","France",0,1,"qf",2006,1,5),
    ("Germany","Italy",0,2,"sf",2006,22,13),("Italy","France",1,1,"final",2006,13,5),
    ("Germany","Saudi Arabia",8,0,"group",2002,11,37),("Korea Republic","Poland",2,0,"group",2002,40,28),
    ("USA","Portugal",3,2,"group",2002,13,10),("Brazil","Turkey",2,1,"group",2002,3,13),
    ("Germany","Paraguay",1,0,"r16",2002,11,30),("England","Denmark",3,0,"r16",2002,10,21),
    ("Korea Republic","Italy",2,1,"r16",2002,40,16),("Brazil","Belgium",2,0,"r16",2002,3,22),
    ("Germany","USA",1,0,"qf",2002,11,13),("Brazil","England",2,1,"qf",2002,3,10),
    ("Germany","Korea Republic",1,0,"sf",2002,11,40),("Brazil","Turkey",1,0,"sf",2002,3,13),
    ("Brazil","Germany",2,0,"final",2002,3,11),
    ("France","Brazil",3,0,"final",1998,18,1),("Netherlands","Argentina",2,1,"qf",1998,6,10),
    ("Brazil","Denmark",3,2,"qf",1998,1,15),("France","Netherlands",2,1,"sf",1998,18,6),
    ("Brazil","Italy",0,0,"final",1994,1,4),("Brazil","Sweden",1,0,"sf",1994,1,13),
    ("Italy","Bulgaria",2,1,"sf",1994,4,31),("West Germany","Argentina",1,0,"final",1990,9,1),
    ("Argentina","Brazil",1,0,"r16",1990,1,2),("West Germany","England",1,1,"sf",1990,9,7),
    ("Argentina","Italy",1,1,"sf",1990,1,4),
]

def main():
    df = pd.DataFrame(MATCHES_RAW,
        columns=['team1','team2','goals1','goals2','phase','year','rank1','rank2'])

    def get_result(row):
        if row['goals1'] > row['goals2']: return 1
        if row['goals1'] < row['goals2']: return 2
        return 0

    df['result']    = df.apply(get_result, axis=1)
    phase_map       = {'group':0,'r16':1,'qf':2,'sf':3,'3rd':3,'final':4}
    df['phase_enc'] = df['phase'].map(phase_map).fillna(0)
    df['rank_diff'] = df['rank1'] - df['rank2']
    df['rank_ratio']= df['rank2'] / (df['rank1'] + 0.01)

    all_t = pd.concat([
        df[['team1','result']].rename(columns={'team1':'team'})
          .assign(win=df['result'].map({1:1,0:0.5,2:0})),
        df[['team2','result']].rename(columns={'team2':'team'})
          .assign(win=df['result'].map({2:1,0:0.5,1:0}))
    ])
    form_rate = all_t.groupby('team')['win'].mean().to_dict()
    df['form_diff'] = df['team1'].map(form_rate).fillna(.38) - df['team2'].map(form_rate).fillna(.38)

    gols = pd.concat([
        df[['team1','goals1']].rename(columns={'team1':'team','goals1':'goals'}),
        df[['team2','goals2']].rename(columns={'team2':'team','goals2':'goals'})
    ])
    avg_goals = gols.groupby('team')['goals'].mean().to_dict()
    df['avg_goals_diff'] = df['team1'].map(avg_goals).fillna(1.2) - df['team2'].map(avg_goals).fillna(1.2)

    titles = {'Brazil':5,'Germany':4,'Italy':4,'France':2,'Argentina':3,
              'Uruguay':2,'England':1,'Spain':1,'West Germany':3}
    df['title_diff'] = df['team1'].map(titles).fillna(0) - df['team2'].map(titles).fillna(0)

    exp = all_t.groupby('team').size().to_dict()
    df['exp_diff'] = df['team1'].map(exp).fillna(3) - df['team2'].map(exp).fillna(3)

    FEATURES = ['rank_diff','rank_ratio','form_diff','avg_goals_diff',
                'title_diff','exp_diff','phase_enc']
    X, y = df[FEATURES], df['result']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, C=0.5, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=4,
            class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42),
    }

    print(f"\nDataset: {len(df)} partidas")
    best_model, best_score, best_name = None, 0, ''
    results_summary = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=cv, scoring='accuracy')
        results_summary[name] = {'mean': round(scores.mean()*100,1), 'std': round(scores.std()*100,1)}
        print(f"  {name}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
        if scores.mean() > best_score:
            best_score, best_model, best_name = scores.mean(), m, name

    best_model.fit(X, y)
    print(f"\n✅  Melhor: {best_name} ({best_score*100:.1f}%)")

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    with open('models/wc_model.pkl','wb') as f: pickle.dump(best_model, f)
    stats = {
        'form_rate':  form_rate, 'avg_goals': avg_goals, 'titles': titles,
        'team_games': {k:int(v) for k,v in exp.items()},
        'best_model': best_name, 'accuracy': round(best_score*100,1),
        'all_models': results_summary, 'features': FEATURES, 'n_matches': len(df),
    }
    with open('data/wc_stats.json','w') as f: json.dump(stats, f, indent=2)
    print("✅  models/wc_model.pkl e data/wc_stats.json salvos")

if __name__ == '__main__':
    main()
