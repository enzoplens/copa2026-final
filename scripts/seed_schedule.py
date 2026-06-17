"""
scripts/seed_schedule.py
========================
Agenda fixa da Copa 2026 (grupos + datas + cidades), usada como
fallback enquanto a API externa não retornou dados ainda — por
exemplo, na primeira execução do workflow, antes do primeiro
fetch bem-sucedido.

Os placares aqui são None para jogos futuros; update.py sobrescreve
com dados reais sempre que a API responde com sucesso. Sem esse
seed, a primeira execução do workflow gera um live.json com
matches: [] e o site fica vazio até a API funcionar.

Os 4 primeiros jogos já têm resultado fixo porque já aconteceram
(dados de 11-12 jun 2026) e servem de verificação visual rápida
de que o pipeline está vivo mesmo sem token configurado.
"""

SEED_MATCHES = [
    {'id':'seed-1','t1':'Mexico','t2':'South Africa','f1':'🇲🇽','f2':'🇿🇦',
     'group':'Grupo A','stage':'GROUP_STAGE','date':'11 jun','time':'—',
     's1':2,'s2':0,'status':'done'},
    {'id':'seed-2','t1':'South Korea','t2':'Czech Republic','f1':'🇰🇷','f2':'🇨🇿',
     'group':'Grupo A','stage':'GROUP_STAGE','date':'11 jun','time':'—',
     's1':2,'s2':1,'status':'done'},
    {'id':'seed-3','t1':'Canada','t2':'Switzerland','f1':'🇨🇦','f2':'🇨🇭',
     'group':'Grupo B','stage':'GROUP_STAGE','date':'12 jun','time':'—',
     's1':1,'s2':0,'status':'done'},
    {'id':'seed-4','t1':'USA','t2':'Paraguay','f1':'🇺🇸','f2':'🇵🇾',
     'group':'Grupo D','stage':'GROUP_STAGE','date':'12 jun','time':'—',
     's1':1,'s2':0,'status':'done'},
    {'id':'seed-5','t1':'Australia','t2':'Turkey','f1':'🇦🇺','f2':'🇹🇷',
     'group':'Grupo D','stage':'GROUP_STAGE','date':'13 jun','time':'01:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-6','t1':'Brazil','t2':'Morocco','f1':'🇧🇷','f2':'🇲🇦',
     'group':'Grupo C','stage':'GROUP_STAGE','date':'13 jun','time':'19:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-7','t1':'Haiti','t2':'Scotland','f1':'🇭🇹','f2':'🏴󠁧󠁢󠁳󠁣󠁴󠁿',
     'group':'Grupo C','stage':'GROUP_STAGE','date':'13 jun','time':'22:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-8','t1':'Germany','t2':'Curacao','f1':'🇩🇪','f2':'🇨🇼',
     'group':'Grupo E','stage':'GROUP_STAGE','date':'14 jun','time':'14:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-9','t1':'Netherlands','t2':'Japan','f1':'🇳🇱','f2':'🇯🇵',
     'group':'Grupo F','stage':'GROUP_STAGE','date':'14 jun','time':'17:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-10','t1':'Spain','t2':'Cape Verde','f1':'🇪🇸','f2':'🇨🇻',
     'group':'Grupo H','stage':'GROUP_STAGE','date':'15 jun','time':'15:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-11','t1':'France','t2':'Senegal','f1':'🇫🇷','f2':'🇸🇳',
     'group':'Grupo I','stage':'GROUP_STAGE','date':'15 jun','time':'22:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-12','t1':'Portugal','t2':'Uzbekistan','f1':'🇵🇹','f2':'🇺🇿',
     'group':'Grupo K','stage':'GROUP_STAGE','date':'16 jun','time':'16:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-13','t1':'Argentina','t2':'Algeria','f1':'🇦🇷','f2':'🇩🇿',
     'group':'Grupo J','stage':'GROUP_STAGE','date':'16 jun','time':'22:00 UTC',
     's1':None,'s2':None,'status':'sched'},
    {'id':'seed-14','t1':'England','t2':'Croatia','f1':'🏴󠁧󠁢󠁥󠁮󠁧󠁿','f2':'🇭🇷',
     'group':'Grupo L','stage':'GROUP_STAGE','date':'17 jun','time':'—',
     's1':None,'s2':None,'status':'sched'},
]
