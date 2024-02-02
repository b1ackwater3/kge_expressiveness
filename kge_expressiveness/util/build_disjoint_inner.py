
from collections import defaultdict
import re
# 读取train typeOf 的train 的数据集

#%%
def read_triples(file_path):
    triples = []
    with open(file_path, encoding='utf-8') as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((h,r,t))
    return triples

train_files = "/home/ai/yl/data/YAGO3-1668k-old/yago_type/train.txt"

typeOfTriples = read_triples(train_files)

old_disjoint = read_triples("/home/ai/yl/data/YAGO3-1668k-old/yago_ontonet/disjoint.txt")
 
t2Instance = defaultdict(set)
for h,r,t in typeOfTriples:
    t2Instance[t].add(h)


disjoint_triples = []

typeKeys = list(t2Instance.keys())

num = len(typeKeys)

for i in range(len(typeKeys)):
    t1 = typeKeys[i]
    for j in range(i+1, len(typeKeys)):
        t2 = typeKeys[j]
        if t1 != t2 and t2Instance[t1].isdisjoint(t2Instance[t2]):
            disjoint_triples.append((t1,"rdfs:disjoint",t2))

old_disjoint = set(old_disjoint)

filtered_disjoint = []
for h,r,t in disjoint_triples:
    if (h,r,t ) not in old_disjoint and (t,r,h) not in old_disjoint:
        filtered_disjoint.append((h,r,t))

#%%
h = "wikicat_Argentine_expatriate_footballers"
if re.match(".*footballers.*",h):
    print("OK")

#%%
print(len(filtered_disjoint))

all_triples = filtered_disjoint
#%%
disjoint_triples = all_triples

filtered_disjoint = []
less_triples = []
for h,r,t in disjoint_triples:
    if re.match(".*footballers.*",h) and re.match(".*footballers.*", t):
        filtered_disjoint.append((h,r,t))
    else:
        less_triples.append((h,r,t))
# 规则1: xxx_footballers 之间是不相互重叠的
# %%
print(len(filtered_disjoint))
meet_players = filtered_disjoint
# %%
# remove player 和 电影之间的disjoint：信息含量不高
print(less_triples[0:10])
disjoint_triples = less_triples
filtered_disjoint = []
less_triples = []
for h,r,t in disjoint_triples:
    if re.match(".*footballers.*",h) and re.match(".*films.*", t):
        filtered_disjoint.append((h,r,t))
    else:
        less_triples.append((h,r,t))
# %%
print(less_triples[0:10])
disjoint_triples = less_triples
filtered_disjoint = []
less_triples = []
for h,r,t in disjoint_triples:
    if (re.match(".*footballers.*",h) or re.match(".*people.*",h) or re.match(".*artist.*",h)) and re.match(".*person.*", t):
        filtered_disjoint.append((h,r,t))
    else:
        less_triples.append((h,r,t))
# %%
disjoint_triples = less_triples
filtered_disjoint = []
less_triples = []
for h,r,t in disjoint_triples:
    if re.match(".*footballers.*",h) or  re.match(".*footballers.*",t)  :
        filtered_disjoint.append((h,r,t))
    else:
        less_triples.append((h,r,t))

all_footballers = filtered_disjoint

# %%
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*film.*",h) or re.match(".*film.*",t):
        continue
    else:
        all_footballers_foot.append((h,r,t))

print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*person.*",h) or re.match(".*person.*",t) \
        or re.match(".*cities_and_towns.*",h) or re.match(".*cities_and_towns.*",t) \
         or re.match(".*_player_110.*",h) or re.match(".*_player_110.*",t)   \
            :
        continue
    else:
        all_footballers_foot.append((h,r,t))
print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))

# %%
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*athlete.*",h) or re.match(".*athlete.*",t) \
        or re.match(".*first_name.*",h) or re.match(".*first_name.*",t) \
        or re.match(".*football_official.*",h) or re.match(".*football_official.*",t)  \
        or re.match(".*song.*",h) or re.match(".*song.*",t) or re.match(".*title.*",h) or re.match(".*title.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))

all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*end.*",h) or re.match(".*end.*",t) \
        or re.match(".*name.*",h) or re.match(".*name.*",t) \
        or re.match(".*computer_game.*",h) or re.match(".*computer_game.*",t)  \
        or re.match(".*biography.*",h) or re.match(".*biography.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*village.*",h) or re.match(".*village.*",t) \
        or re.match(".*coin.*",h) or re.match(".*coin.*",t) \
        or re.match(".*lama.*",h) or re.match(".*lama.*",t)  \
        or re.match(".*serial_killer.*",h) or re.match(".*serial_killer.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*sea.*",h) or re.match(".*sea.*",t) \
        or re.match(".*election.*",h) or re.match(".*election.*",t) \
        or re.match(".*sex.*",h) or re.match(".*sex.*",t)  \
        or re.match(".*happening.*",h) or re.match(".*happening.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))

all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*comic_strip.*",h) or re.match(".*comic_strip.*",t) \
        or re.match(".*hill.*",h) or re.match(".*hill.*",t) \
        or re.match(".*war.*",h) or re.match(".*war.*",t)  \
        or re.match(".*valley.*",h) or re.match(".*valley.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*treaty.*",h) or re.match(".*treaty.*",t) \
        or re.match(".*attack.*",h) or re.match(".*attack.*",t) \
        or re.match(".*center.*",h) or re.match(".*center.*",t)  \
        or re.match(".*mayor.*",h) or re.match(".*mayor.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*cabal.*",h) or re.match(".*cabal.*",t) \
        or re.match(".*bay.*",h) or re.match(".*bay.*",t) \
        or re.match(".*lake.*",h) or re.match(".*lake.*",t)  \
        or re.match(".*language.*",h) or re.match(".*language.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))

all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*order.*",h) or re.match(".*order.*",t) \
        or re.match(".*zone.*",h) or re.match(".*zone.*",t) \
        or re.match(".*towns.*",h) or re.match(".*towns.*",t)  \
        or re.match(".*economist.*",h) or re.match(".*economist.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))

all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*convention.*",h) or re.match(".*convention.*",t) \
        or re.match(".*sacred_text.*",h) or re.match(".*sacred_text.*",t) \
        or re.match(".*school_district.*",h) or re.match(".*school_district.*",t)  \
        or re.match(".*trail.*",h) or re.match(".*trail.*",t) or re.match(".*contest.*",h) or re.match(".*contest.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))

print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))

#%%
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*sport.*",h) or re.match(".*sport.*",t) \
        or re.match(".*state.*",h) or re.match(".*state.*",t) \
        or re.match(".*channel.*",h) or re.match(".*channel.*",t)  \
        or re.match(".*park.*",h) or re.match(".*park.*",t) or re.match(".*event.*",h) or re.match(".*event.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
all_footballers = all_footballers_foot
all_footballers_foot = []
for h,r,t in all_footballers:
    if re.match(".*scholar.*",h) or re.match(".*scholar.*",t) \
        or re.match(".*peninsula.*",h) or re.match(".*peninsula.*",t) \
        or re.match(".*channel.*",h) or re.match(".*channel.*",t)  \
        or re.match(".*park.*",h) or re.match(".*park.*",t) or re.match(".*event.*",h) or re.match(".*event.*",t) :
        continue
    else:
        all_footballers_foot.append((h,r,t))
print(len(all_footballers_foot))
with open("./player.txt", "w+") as f:
    for h,r,t in all_footballers_foot:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
with open("./others.txt", "w+") as f:
    for h,r,t in less_triples:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
all_footballers = less_triples
less_triples = []
for h,r,t in all_footballers:
    if re.match(".*administrative_district.*",h) or re.match(".*administrative_district.*",t) :
        continue
    else:
        less_triples.append((h,r,t))

with open("./others.txt", "w+") as f:
    for h,r,t in less_triples:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
all_footballers = less_triples
less_triples = []
for h,r,t in all_footballers:
    if (re.match(".*player.*",h) or re.match(".*artist.*",h)) and (re.match(".*film.*",t) or re.match(".*town.*",t) or  re.match(".*movie.*",t)  or  re.match(".*county.*",t) or  re.match(".*Film.*",t)) :
        continue
    else:
        less_triples.append((h,r,t))

all_footballers = less_triples
less_triples = []
for h,r,t in all_footballers:
    if re.match(".*film.*",h) or re.match(".*film.*",t) :
        continue
    else:
        less_triples.append((h,r,t))
all_footballers = less_triples
less_triples = []
for h,r,t in all_footballers:
    if re.match(".*wordnet_person.*",h) or re.match(".*wordnet_person.*",t) :
        continue
    else:
        less_triples.append((h,r,t))

with open("./others.txt", "w+") as f:
    for h,r,t in less_triples:
        f.write("%s\t%s\t%s\n" % (h,r,t))
# %%
print(less_triples)